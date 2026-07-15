#ifndef SHARED_MEMORY_UTILS_H_
#define SHARED_MEMORY_UTILS_H_

#include <pthread.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <system_error>

namespace batchgen::kv {

enum class SharedMemoryInitState : std::uint32_t {
    kUninitialized = 0,
    kInitializing = 1,
    kReady = 2,
};

inline std::size_t AlignUp(std::size_t value, std::size_t alignment) {
    if (alignment == 0) {
        return value;
    }
    const std::size_t remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    return value + (alignment - remainder);
}

inline std::size_t SystemPageSize() {
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        throw std::system_error(errno, std::generic_category(),
                                "sysconf(_SC_PAGESIZE) failed");
    }
    return static_cast<std::size_t>(page_size);
}

class ScopedPthreadMutexLock {
   public:
    explicit ScopedPthreadMutexLock(pthread_mutex_t* mutex) : mutex_(mutex) {
        const int rc = pthread_mutex_lock(mutex_);
        if (rc == EOWNERDEAD) {
            const int consistent_rc = pthread_mutex_consistent(mutex_);
            if (consistent_rc != 0) {
                throw std::system_error(consistent_rc, std::generic_category(),
                                        "pthread_mutex_consistent failed");
            }
        } else if (rc != 0) {
            throw std::system_error(rc, std::generic_category(),
                                    "pthread_mutex_lock failed");
        }
    }

    ScopedPthreadMutexLock(const ScopedPthreadMutexLock&) = delete;
    ScopedPthreadMutexLock& operator=(const ScopedPthreadMutexLock&) = delete;

    ~ScopedPthreadMutexLock() {
        const int rc = pthread_mutex_unlock(mutex_);
        if (rc != 0) {
            std::terminate();
        }
    }

   private:
    pthread_mutex_t* mutex_;
};

inline void InitProcessSharedRobustMutex(pthread_mutex_t* mutex,
                                         const char* name) {
    pthread_mutexattr_t attr;
    if (const int rc = pthread_mutexattr_init(&attr); rc != 0) {
        throw std::system_error(rc, std::generic_category(),
                                "pthread_mutexattr_init failed");
    }
    if (const int rc =
            pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
        rc != 0) {
        pthread_mutexattr_destroy(&attr);
        throw std::system_error(rc, std::generic_category(),
                                "pthread_mutexattr_setpshared failed");
    }
    if (const int rc = pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);
        rc != 0) {
        pthread_mutexattr_destroy(&attr);
        throw std::system_error(rc, std::generic_category(),
                                "pthread_mutexattr_setrobust failed");
    }
    if (const int rc = pthread_mutex_init(mutex, &attr); rc != 0) {
        pthread_mutexattr_destroy(&attr);
        throw std::system_error(rc, std::generic_category(), name);
    }
    pthread_mutexattr_destroy(&attr);
}

}  // namespace batchgen::kv

#endif  // SHARED_MEMORY_UTILS_H_
