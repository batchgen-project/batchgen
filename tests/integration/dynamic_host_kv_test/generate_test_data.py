"""Generate test datasets for dynamic host KV cache reservation testing.

Creates JSONL files in OpenAI batch format for scenarios A-F, using the
GPT-OSS tiktoken tokenizer for exact token counting.

Usage:
    python generate_test_data.py --scenario A --output data/scenarioA.jsonl
    python generate_test_data.py --scenario all --output-dir data/
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Add BatchGen to path for tokenizer import
BATCHGEN_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BATCHGEN_ROOT))

MODEL_NAME = "openai/gpt-oss-120b"


# ============ Synthetic Text Generation ============

TOPIC_PARAGRAPHS = {
    "physics": (
        "The fundamental forces of nature govern all physical interactions in the universe. "
        "Gravity, the weakest of the four forces, dominates at astronomical scales due to its "
        "infinite range and always-attractive nature. Einstein's general relativity describes "
        "gravity as the curvature of spacetime caused by mass and energy. At quantum scales, "
        "the electromagnetic force binds electrons to nuclei, while the strong nuclear force "
        "confines quarks within protons and neutrons. The weak nuclear force mediates radioactive "
        "decay and neutrino interactions. Attempts to unify these forces into a single framework "
        "remain one of physics' greatest challenges. String theory proposes that fundamental "
        "particles are one-dimensional vibrating strings, while loop quantum gravity attempts "
        "to quantize spacetime itself. Both approaches predict phenomena at the Planck scale "
        "that are currently beyond experimental reach."
    ),
    "biology": (
        "Cellular biology reveals the intricate machinery of life at its most fundamental level. "
        "The central dogma of molecular biology describes how genetic information flows from DNA "
        "through RNA to proteins. Transcription factors regulate gene expression by binding to "
        "specific DNA sequences called promoters and enhancers. Epigenetic modifications, such "
        "as DNA methylation and histone acetylation, can alter gene expression without changing "
        "the underlying DNA sequence. The discovery of CRISPR-Cas9 gene editing has revolutionized "
        "biology by enabling precise modifications to any organism's genome. Mitochondria, the "
        "powerhouses of the cell, contain their own circular DNA and are thought to have originated "
        "from ancient endosymbiotic bacteria. The human microbiome contains trillions of bacteria "
        "that play crucial roles in digestion, immunity, and even mental health."
    ),
    "history": (
        "The Industrial Revolution fundamentally transformed human society beginning in late "
        "18th century Britain. The transition from agrarian economies to industrial manufacturing "
        "was driven by technological innovations including the steam engine, spinning jenny, and "
        "power loom. Urbanization accelerated as workers migrated to factory towns, creating both "
        "unprecedented wealth and severe social problems. Child labor, dangerous working conditions, "
        "and urban poverty led to reform movements and eventually labor legislation. The revolution "
        "spread from Britain to continental Europe, North America, and eventually the world. "
        "The Second Industrial Revolution of the late 19th century introduced electricity, the "
        "internal combustion engine, and new chemical processes. These changes laid the groundwork "
        "for the modern global economy and continue to shape political and social structures today."
    ),
    "technology": (
        "Artificial intelligence has progressed from narrow task-specific systems to increasingly "
        "general-purpose models. Early AI research focused on symbolic reasoning and expert systems, "
        "while modern approaches rely heavily on deep learning and neural networks. Transformer "
        "architectures, introduced in 2017, revolutionized natural language processing by enabling "
        "parallel processing of sequential data through self-attention mechanisms. Large language "
        "models trained on vast text corpora demonstrate emergent capabilities including reasoning, "
        "code generation, and creative writing. GPU computing and distributed training systems "
        "have enabled training models with hundreds of billions of parameters. Challenges remain "
        "in areas such as hallucination reduction, alignment with human values, and energy "
        "efficiency. Edge deployment and model compression techniques aim to bring AI capabilities "
        "to resource-constrained devices."
    ),
    "mathematics": (
        "Number theory explores the properties and relationships of integers, forming one of the "
        "oldest branches of mathematics. The Riemann Hypothesis, proposed in 1859, concerns the "
        "distribution of prime numbers and remains one of the most important unsolved problems. "
        "Fermat's Last Theorem, stating that no three positive integers satisfy a^n + b^n = c^n "
        "for any integer n greater than 2, was proved by Andrew Wiles in 1995 after 358 years. "
        "Modular arithmetic, which studies remainders after division, has applications in "
        "cryptography, computer science, and music theory. The theory of elliptic curves connects "
        "algebraic geometry with number theory and played a crucial role in Wiles' proof. "
        "Computational number theory uses algorithms to study prime factorization, a problem "
        "whose difficulty underlies the security of RSA encryption."
    ),
    "chemistry": (
        "Organic chemistry studies carbon-containing compounds and their reactions, forming the "
        "basis of biochemistry and pharmaceutical science. Carbon's ability to form four stable "
        "covalent bonds enables the vast diversity of organic molecules. Functional groups such "
        "as hydroxyl, carboxyl, amino, and phosphate groups determine the chemical properties "
        "and reactivity of organic compounds. Catalysis, both homogeneous and heterogeneous, "
        "enables chemical reactions to proceed at lower temperatures and with greater selectivity. "
        "Green chemistry principles aim to reduce waste and minimize environmental impact through "
        "atom-efficient reactions and renewable feedstocks. Supramolecular chemistry explores "
        "non-covalent interactions such as hydrogen bonding, van der Waals forces, and "
        "pi-pi stacking to create complex self-assembled structures."
    ),
    "geography": (
        "Plate tectonics describes the large-scale motion of Earth's lithosphere, explaining "
        "phenomena from earthquakes to mountain formation. The lithosphere is divided into major "
        "and minor tectonic plates that float on the partially molten asthenosphere. Convergent "
        "boundaries produce subduction zones and volcanic arcs, while divergent boundaries create "
        "mid-ocean ridges and rift valleys. Transform boundaries, where plates slide past each "
        "other, produce strike-slip faults like the San Andreas Fault. The theory of plate "
        "tectonics unified previously disparate observations about continental drift, seafloor "
        "spreading, and the distribution of fossils across continents. Mantle convection, driven "
        "by heat from Earth's core, provides the energy that drives plate motion. Hotspot volcanism "
        "creates island chains like Hawaii as plates move over stationary magma plumes."
    ),
    "economics": (
        "Macroeconomic theory examines the behavior of economies as a whole, including output, "
        "unemployment, and inflation. Keynesian economics emphasizes the role of aggregate demand "
        "in determining economic output, arguing that government intervention through fiscal and "
        "monetary policy can stabilize business cycles. Monetarist theory, associated with Milton "
        "Friedman, focuses on the money supply as the primary determinant of economic activity. "
        "Modern monetary theory challenges conventional views about government debt, arguing that "
        "sovereign currency issuers cannot become insolvent in their own currency. Behavioral "
        "economics incorporates psychological insights into economic models, demonstrating that "
        "humans systematically deviate from the rational actor model. International trade theory "
        "examines how comparative advantage and economies of scale drive global commerce."
    ),
}

NEEDLE_FACTS = [
    {
        "topic": "quantum entanglement",
        "fact": (
            "CRITICAL FACT: In the landmark 2023 Zurich experiment, researchers demonstrated "
            "quantum entanglement across a distance of exactly 1,247 kilometers using photon "
            "pairs generated at a rate of 3.7 million per second, achieving a Bell inequality "
            "violation of 4.23 standard deviations, definitively ruling out local hidden "
            "variable theories at the macroscopic scale."
        ),
    },
    {
        "topic": "deep sea discovery",
        "fact": (
            "CRITICAL FACT: The 2024 Mariana Expedition discovered a previously unknown species "
            "of bioluminescent cephalopod at 8,912 meters depth, designated Vampyroteuthis "
            "profundissimus, which produces light at wavelength 472nm and has a tentacle span "
            "of exactly 2.3 meters, making it the largest known deep-sea cephalopod."
        ),
    },
    {
        "topic": "archaeological find",
        "fact": (
            "CRITICAL FACT: Excavations at the Gobekli Tepe site in 2024 uncovered a previously "
            "unknown underground chamber containing 47 carved limestone pillars dating to "
            "approximately 11,600 BCE, each depicting a unique astronomical constellation "
            "pattern that predates known star charts by at least 5,000 years."
        ),
    },
    {
        "topic": "materials science",
        "fact": (
            "CRITICAL FACT: A team at MIT in 2024 synthesized a room-temperature superconductor "
            "with the chemical formula Lu3H10N at a pressure of 1.2 GPa, exhibiting zero "
            "resistance below 294K and a critical magnetic field of 12.7 Tesla, with the "
            "crystal structure confirmed by X-ray diffraction to be a modified clathrate."
        ),
    },
    {
        "topic": "renewable energy",
        "fact": (
            "CRITICAL FACT: The Saharan Solar Grid project completed in 2025 achieved a peak "
            "output of 847 gigawatts from 12.4 million concentrated solar power mirrors "
            "covering exactly 41,200 square kilometers, with an energy storage capacity of "
            "2,340 gigawatt-hours using molten salt thermal storage at 565 degrees Celsius."
        ),
    },
]


def _get_tokenizer():
    """Load GPT-OSS tiktoken tokenizer.

    Tries to import the full GPTOssTokenizer first. Falls back to creating
    a compatible tiktoken encoding directly (avoids transformers dependency).
    """
    try:
        from batchgen.models.openai.gpt_oss_120b.tokenizer import GPTOssTokenizer
        return GPTOssTokenizer()
    except (ImportError, ModuleNotFoundError):
        pass

    # Fallback: create tiktoken o200k_harmony directly (same as GPTOssTokenizer)
    import tiktoken
    logger.info("Using tiktoken o200k_base directly (no transformers needed)")
    return tiktoken.get_encoding("o200k_base")


def _count_tokens(tokenizer, text: str) -> int:
    """Count tokens in text using the tokenizer."""
    if hasattr(tokenizer, 'encode'):
        result = tokenizer.encode(text)
        if isinstance(result, list):
            return len(result)
        return len(result)
    return len(tokenizer.encode(text))


def _make_request(custom_id: str, messages: List[Dict], max_tokens: int) -> dict:
    """Create a single JSONL request in OpenAI batch format."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
    }


def _build_long_document(tokenizer, target_tokens: int, needle_facts: List[dict],
                         seed: int = 42) -> Tuple[str, List[str]]:
    """Build a document of ~target_tokens with embedded needle facts.

    Returns (document_text, list_of_needle_topics).
    """
    rng = random.Random(seed)
    topics = list(TOPIC_PARAGRAPHS.keys())
    paragraphs = []
    section_num = 1

    # Pre-build sections
    while True:
        topic = topics[section_num % len(topics)]
        para = f"\n\n--- Section {section_num}: {topic.title()} ---\n\n{TOPIC_PARAGRAPHS[topic]}"
        paragraphs.append(para)
        section_num += 1
        # Check if we have enough
        current = "\n".join(paragraphs)
        count = _count_tokens(tokenizer, current)
        if count >= target_tokens + 5000:  # Overshoot, then trim
            break

    # Insert needle facts at random positions
    needle_topics = []
    fact_indices = sorted(rng.sample(range(len(paragraphs)), min(len(needle_facts), len(paragraphs))))
    for i, fact_idx in enumerate(fact_indices):
        if i < len(needle_facts):
            needle = needle_facts[i]
            needle_topics.append(needle["topic"])
            paragraphs.insert(fact_idx + i, f"\n\n{needle['fact']}\n")

    # Binary search to trim to exact target
    full_text = "\n".join(paragraphs)
    tokens = tokenizer.encode(full_text) if not hasattr(tokenizer, 'tokenizer') else tokenizer.encode(full_text)
    if isinstance(tokens, list) and len(tokens) > target_tokens:
        # Decode back to exactly target_tokens
        trimmed = tokens[:target_tokens]
        if hasattr(tokenizer, 'decode'):
            full_text = tokenizer.decode(trimmed)
        elif hasattr(tokenizer, 'tokenizer'):
            full_text = tokenizer.tokenizer.decode(trimmed)

    final_count = _count_tokens(tokenizer, full_text)
    logger.info(f"Built document: {final_count} tokens (target: {target_tokens})")

    return full_text, needle_topics


# ============ Scenario Generators ============

def generate_scenario_a(tokenizer, output_path: Path):
    """Short decode baseline — 64 sequences, ~200-500 token prompts, max_tokens=128."""
    questions = [
        "Explain the concept of gravitational waves and how they were first detected.",
        "What are the main differences between classical and quantum computing?",
        "Describe the process of photosynthesis in detail.",
        "How does the TCP/IP protocol stack work?",
        "Explain the significance of the Turing test in artificial intelligence.",
        "What caused the fall of the Roman Empire?",
        "Describe the structure and function of DNA.",
        "How do neural networks learn from data?",
        "Explain the theory of plate tectonics.",
        "What are the key principles of object-oriented programming?",
        "Describe the water cycle and its importance to Earth's climate.",
        "How does public key cryptography ensure secure communication?",
        "Explain the economic concept of supply and demand.",
        "What is the role of mitochondria in cellular respiration?",
        "Describe the main features of Renaissance art.",
        "How does a compiler translate source code to machine code?",
    ]

    requests = []
    for i in range(64):
        q = questions[i % len(questions)]
        # Add context to vary prompt length (200-500 tokens)
        context = f"Question {i + 1} of 64. Please provide a clear, concise answer. "
        topic = list(TOPIC_PARAGRAPHS.keys())[i % len(TOPIC_PARAGRAPHS)]
        padding = TOPIC_PARAGRAPHS[topic][:200]  # Short padding
        prompt = f"{context}\n\nContext: {padding}\n\nQuestion: {q}"

        token_count = _count_tokens(tokenizer, prompt)
        messages = [
            {"role": "system", "content": "You are a knowledgeable assistant. Answer concisely."},
            {"role": "user", "content": prompt},
        ]
        requests.append(_make_request(f"scenA-{i}", messages, max_tokens=128))

    _write_jsonl(requests, output_path)
    logger.info(f"Scenario A: {len(requests)} sequences, ~200-500 token prompts, max_tokens=128")


def generate_scenario_b(tokenizer, output_path: Path):
    """Single growth — 32 sequences, ~500 token prompts, max_tokens=8192."""
    requests = []
    topics = list(TOPIC_PARAGRAPHS.keys())

    for i in range(32):
        topic = topics[i % len(topics)]
        para = TOPIC_PARAGRAPHS[topic]
        prompt = (
            f"You have been given the following passage about {topic}:\n\n"
            f"{para}\n\n"
            f"Based on this passage, write a comprehensive essay expanding on every point "
            f"mentioned. Include additional context, examples, and analysis for each concept."
        )
        token_count = _count_tokens(tokenizer, prompt)
        messages = [
            {"role": "system", "content": "You are an expert writer. Produce detailed, thorough essays."},
            {"role": "user", "content": prompt},
        ]
        requests.append(_make_request(f"scenB-{i}", messages, max_tokens=8192))

    _write_jsonl(requests, output_path)
    logger.info(f"Scenario B: {len(requests)} sequences, ~500 token prompts, max_tokens=8192")


def generate_scenario_c(tokenizer, output_path: Path):
    """Multi growth — 16 sequences, ~500 token prompts, max_tokens=32768."""
    requests = []
    topics = list(TOPIC_PARAGRAPHS.keys())

    for i in range(16):
        topic = topics[i % len(topics)]
        para = TOPIC_PARAGRAPHS[topic]
        prompt = (
            f"You have been given the following passage about {topic}:\n\n"
            f"{para}\n\n"
            f"Write an extremely detailed, book-chapter-length analysis of this topic. "
            f"Cover the history, current state, future directions, key debates, notable "
            f"figures, landmark discoveries, practical applications, and philosophical "
            f"implications. Be as thorough and comprehensive as possible."
        )
        messages = [
            {"role": "system", "content": "You are a world-class academic writer. Write at extreme length and depth."},
            {"role": "user", "content": prompt},
        ]
        requests.append(_make_request(f"scenC-{i}", messages, max_tokens=32768))

    _write_jsonl(requests, output_path)
    logger.info(f"Scenario C: {len(requests)} sequences, ~500 token prompts, max_tokens=32768")


def generate_scenario_d(tokenizer, output_path: Path):
    """64K+64K — 4 sequences, ~64K token prompts, max_tokens=65536."""
    target_prompt_tokens = 64000
    requests = []

    # Each sequence gets a different subset of needle facts
    rng = random.Random(42)
    for i in range(4):
        # Pick 3-5 needle facts for this sequence
        n_needles = rng.randint(3, 5)
        seq_needles = rng.sample(NEEDLE_FACTS, n_needles)
        doc_text, needle_topics = _build_long_document(
            tokenizer, target_prompt_tokens, seq_needles, seed=42 + i
        )
        question = (
            f"You have just read a very long document. Within it, there are several "
            f"CRITICAL FACTS marked clearly. Find ALL of them and for each one:\n"
            f"1. Quote the fact exactly as stated\n"
            f"2. Explain its significance in detail\n"
            f"3. Discuss how it relates to the surrounding text\n"
            f"4. Analyze potential implications\n\n"
            f"Be extremely thorough. Write a detailed analysis for each critical fact found."
        )
        full_prompt = f"{doc_text}\n\n{question}"
        actual_tokens = _count_tokens(tokenizer, full_prompt)
        logger.info(f"Scenario D seq {i}: {actual_tokens} prompt tokens, needles: {needle_topics}")

        messages = [
            {"role": "system", "content": "You are a meticulous reader and analyst. Find and analyze all critical facts in the document."},
            {"role": "user", "content": full_prompt},
        ]
        requests.append(_make_request(f"scenD-{i}", messages, max_tokens=65536))

    _write_jsonl(requests, output_path)
    logger.info(f"Scenario D: {len(requests)} sequences, ~64K token prompts, max_tokens=65536")


def generate_scenario_e(tokenizer, output_path: Path):
    """Eviction stress — 64 sequences, ~500 token prompts, max_tokens=8192.

    Same data format as Scenario B but more sequences. Meant to be run with
    small --host-kv-cache-size (32GB) to force eviction.
    """
    requests = []
    topics = list(TOPIC_PARAGRAPHS.keys())

    for i in range(64):
        topic = topics[i % len(topics)]
        para = TOPIC_PARAGRAPHS[topic]
        prompt = (
            f"You have been given the following passage about {topic}:\n\n"
            f"{para}\n\n"
            f"Based on this passage, write a comprehensive essay expanding on every point "
            f"mentioned. Include additional context, examples, and analysis for each concept."
        )
        messages = [
            {"role": "system", "content": "You are an expert writer. Produce detailed, thorough essays."},
            {"role": "user", "content": prompt},
        ]
        requests.append(_make_request(f"scenE-{i}", messages, max_tokens=8192))

    _write_jsonl(requests, output_path)
    logger.info(f"Scenario E: {len(requests)} sequences, ~500 token prompts, max_tokens=8192")


def generate_scenario_f(tokenizer, output_path: Path):
    """Mixed workload — 84 sequences: 64 short + 16 medium + 4 long."""
    requests = []
    topics = list(TOPIC_PARAGRAPHS.keys())
    questions = [
        "Explain the concept of gravitational waves.",
        "What are the differences between RNA and DNA?",
        "How does machine learning differ from traditional programming?",
        "Describe the causes of World War I.",
    ]

    # 64 short sequences (prompt ~200, max_tokens=128)
    for i in range(64):
        q = questions[i % len(questions)]
        prompt = f"Answer briefly: {q}"
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user", "content": prompt},
        ]
        requests.append(_make_request(f"mixS-{i}", messages, max_tokens=128))

    # 16 medium sequences (prompt ~500, max_tokens=4096)
    for i in range(16):
        topic = topics[i % len(topics)]
        para = TOPIC_PARAGRAPHS[topic]
        prompt = f"Expand on this topic:\n\n{para}\n\nWrite a detailed essay."
        messages = [
            {"role": "system", "content": "You are an expert writer."},
            {"role": "user", "content": prompt},
        ]
        requests.append(_make_request(f"mixM-{i}", messages, max_tokens=4096))

    # 4 long sequences (prompt ~500, max_tokens=32768)
    for i in range(4):
        topic = topics[i % len(topics)]
        para = TOPIC_PARAGRAPHS[topic]
        prompt = (
            f"Write a book-chapter-length analysis of {topic}:\n\n{para}\n\n"
            f"Be extremely comprehensive."
        )
        messages = [
            {"role": "system", "content": "You are a world-class academic writer."},
            {"role": "user", "content": prompt},
        ]
        requests.append(_make_request(f"mixL-{i}", messages, max_tokens=32768))

    _write_jsonl(requests, output_path)
    logger.info(
        f"Scenario F: {len(requests)} sequences "
        f"(64 short/128, 16 medium/4096, 4 long/32768)"
    )


def _write_jsonl(requests: list, output_path: Path):
    """Write requests to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(requests)} requests to {output_path}")


SCENARIO_GENERATORS = {
    "A": generate_scenario_a,
    "B": generate_scenario_b,
    "C": generate_scenario_c,
    "D": generate_scenario_d,
    "E": generate_scenario_e,
    "F": generate_scenario_f,
}


def main():
    parser = argparse.ArgumentParser(description="Generate test data for dynamic host KV testing")
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=list(SCENARIO_GENERATORS.keys()) + ["all"],
        help="Which scenario to generate (A-F or 'all')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path (for single scenario)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (for --scenario all)",
    )
    args = parser.parse_args()

    logger.info("Loading tokenizer...")
    tokenizer = _get_tokenizer()
    logger.info("Tokenizer loaded.")

    if args.scenario == "all":
        output_dir = Path(args.output_dir or "data")
        for scenario_id, gen_fn in SCENARIO_GENERATORS.items():
            output_path = output_dir / f"scenario{scenario_id}.jsonl"
            logger.info(f"\n=== Generating Scenario {scenario_id} ===")
            gen_fn(tokenizer, output_path)
    else:
        output_path = Path(args.output or f"data/scenario{args.scenario}.jsonl")
        SCENARIO_GENERATORS[args.scenario](tokenizer, output_path)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
