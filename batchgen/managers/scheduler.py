import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from batchgen.server_args import ServerArgs
import multiprocessing as mp
from transformers import AutoTokenizer

from .batch_schema import BatchObject, BatchStatus
from .completion_schema import (
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRole,
    FinishReason,
    CompletionUsage
)
from .storage import StorageManager

logger = logging.getLogger(__name__)

class ServerScheduler:
    def __init__(self, server_args: ServerArgs):
        '''
        Holds the requests and dispatch to workers.
        
        For multi-node setup (nnodes > 1), this also manages the ZMQ broadcaster
        for synchronizing configuration and task distribution across nodes.
        '''
        self.server_args = server_args
        self.broadcaster = None
        
        self.storage = StorageManager(server_args)
        
        # Initialize broadcaster for multi-node setup
        if server_args.nnodes > 1:
            logger.info(
                f"Initializing ZMQ broadcaster for multi-node setup "
                f"(rank={server_args.node_rank}, nnodes={server_args.nnodes})"
            )
            self.broadcaster = server_args.create_broadcaster()
            logger.info("ZMQ broadcaster initialized successfully")
        else:
            logger.info("Single node setup - broadcaster not needed")
    
    def broadcast(self, obj):
        '''
        Broadcast an object to all nodes in multi-node setup.
        
        Args:
            obj: Object to broadcast (will be pickled)
            
        Returns:
            The broadcast object (for rank 0) or received object (for other ranks)
        '''
        if self.broadcaster is None:
            # Single node setup - just return the object
            return obj
        
        if self.server_args.node_rank == 0:
            logger.info(f"Broadcasting object to {len(self.broadcaster)} nodes")
            return self.broadcaster.broadcast(obj)
        else:
            # Non-zero ranks receive the broadcast
            logger.info("Waiting for broadcast from rank 0")
            return self.broadcaster.broadcast(None)
    
    def close(self):
        '''
        Close the scheduler and cleanup resources.
        '''
        if self.broadcaster is not None:
            logger.info("Closing ZMQ broadcaster")
            self.broadcaster.close()
            self.broadcaster = None

    @classmethod
    def create_batch_queries(cls, file: str):
        '''
        Create batch queries from a given file.
        
        Args:
            file: Path to the input file
        '''
        
        with open(file, 'r') as f:
            firstline = f.readline().strip()
            body = json.loads(firstline)["body"]
            model = body["model"]
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        queries = []
        with open(file, 'r') as f:
            for line in f:
                if line.strip():
                    json_line = line.strip()
                    # Process the json_line to create batch queries
                    body = json.loads(json_line)["body"]
                    model = body["model"]
                    messages = body["messages"]
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    queries.append(text)

        return queries

    def __call__(self, stop_event=None):
        """
        Main loop for processing batches.
        
        Args:
            stop_event: Optional multiprocessing Event to signal shutdown
        """
        while not stop_event.is_set():
            time.sleep(1)
            if self.storage.has_pending_batch():
                batch, input_file_path = self.storage.get_next_pending_batch()
                if batch is not None and input_file_path is not None:
                    queries = self.create_batch_queries(input_file_path)
                    self.run(batch, input_file_path, queries)

    def run(self, batch: BatchObject, input_file_path: Path, queries: list[str]):
        """
        Run batch processing for the given queries.
        
        Args:
            batch: The batch object to process
            input_file_path: Path to the input batch file
            queries: List of formatted query strings to process
        """
        logger.info(f"Starting batch {batch.id} with {len(queries)} queries")
        
        # Update batch status to in_progress
        now = int(datetime.now().timestamp())
        batch.status = BatchStatus.IN_PROGRESS
        batch.in_progress_at = now
        batch.request_counts.total = len(queries)
        self.storage.save_batch(batch)
        logger.info(f"Batch {batch.id} status updated to IN_PROGRESS")
        
        # Mock processing with 1 second sleep
        logger.info(f"Processing {len(queries)} queries (mocking with 1s sleep)...")
        
        time.sleep(1)
        
        # Update batch status to finalizing
        now = int(datetime.now().timestamp())
        batch.status = BatchStatus.FINALIZING
        batch.finalizing_at = now
        batch.request_counts.completed = len(queries)  # Mock all as completed
        self.storage.save_batch(batch)
        logger.info(f"Batch {batch.id} status updated to FINALIZING")
        
        # Write output file
        logger.info(f"Writing output file for batch {batch.id}")
        output_file_id = self.storage.write_batch_output(batch.id, input_file_path, queries)
        batch.output_file_id = output_file_id
        logger.info(f"Output file {output_file_id} created successfully")
        
        # Update batch status to completed
        now = int(datetime.now().timestamp())
        batch.status = BatchStatus.COMPLETED
        batch.completed_at = now
        self.storage.save_batch(batch)
        logger.info(f"Batch {batch.id} completed successfully")
        
        return batch

        