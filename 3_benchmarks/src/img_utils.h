#ifndef IMG_UTILS_WIN_H
#define IMG_UTILS_WIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Windows/MSVC-only implementation: requires <wchar.h>, <io.h>, <fcntl.h> */
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>
#include <io.h>
#include <fcntl.h>

/*
 * Loads an entire file into memory.
 * Parameters:
 *   path      - UTF-16 path to the file (Windows native)
 *   img       - [out] *img will point to a malloc'd buffer with file contents
 *   img_size  - [out] size of the buffer in bytes
 * Returns:
 *   0 on success; 1 open fail; 2 alloc fail; 3 read fail; 4 size overflow.
 */
static inline int img_load(const wchar_t *path, unsigned char **img, size_t *img_size)
{
    if (!path || !img || !img_size) return 1;

    FILE *file = _wfopen(path, L"rb");
    if (!file) return 1;

    if (_fseeki64(file, 0, SEEK_END) != 0) {
        fclose(file); return 1;
    }
    __int64 sz = _ftelli64(file);
    if (sz < 0) {
        fclose(file); return 1;
    }
    if (_fseeki64(file, 0, SEEK_SET) != 0) {
        fclose(file); return 1;
    }

    /* ensure size fits in size_t */
    if (sz > (__int64)SIZE_MAX) {
        fclose(file); return 4;
    }
    *img_size = (size_t)sz;

    *img = (unsigned char*)malloc(*img_size ? *img_size : 1);
    if (!*img) {
        fclose(file); return 2;
    }

    size_t nread = fread(*img, 1, *img_size, file);
    if (nread != *img_size) {
        free(*img);
        *img = NULL;
        fclose(file);
        return 3;
    }

    fclose(file);
    return 0;
}

/*
 * Saves a buffer to a file.
 * Parameters:
 *   path      - UTF-16 path to the output file
 *   img       - data to write (not modified)
 *   img_size  - number of bytes to write
 * Returns:
 *   0 on success; 1 open fail; 2 write fail.
 */
static inline int img_save(const wchar_t *path, const unsigned char *img, size_t img_size)
{
    if (!path || (!img && img_size != 0)) return 1;

    FILE *file = _wfopen(path, L"wb");
    if (!file) return 1;

    size_t nwritten = (img_size ? fwrite(img, 1, img_size, file) : 0);
    if (nwritten != img_size) {
        fclose(file); return 2;
    }

    if (fflush(file) != 0) {
        fclose(file); return 2;
    }

    fclose(file);
    return 0;
}

/*
 * Frees a previously allocated image buffer.
 * Returns:
 *   0 on success; 1 on NULL pointer.
 */
static inline int img_destroy(void *img)
{
    if (!img) return 1;
    free(img);
    return 0;
}

/*
 * Reads all bytes from STDIN (binary mode) into a malloc'd buffer.
 * Parameters:
 *   data  - [out] buffer with data (malloc)
 *   size  - [out] number of bytes read
 * Returns:
 *   0 on success; 1 on allocation failure or I/O error.
 */
static inline int img_load_stdin(unsigned char **data, size_t *size)
{
    if (!data || !size) return 1;

    /* Ensure binary mode to avoid CRLF translation */
    _setmode(_fileno(stdin), _O_BINARY);

    size_t capacity = 4096, length = 0;
    unsigned char *buffer = (unsigned char*)malloc(capacity);
    if (!buffer) return 1;

    for (;;) {
        if (length == capacity) {
            size_t new_cap = capacity < (SIZE_MAX / 2) ? (capacity * 2) : SIZE_MAX;
            if (new_cap == capacity) { free(buffer); return 1; }
            unsigned char *tmp = (unsigned char*)realloc(buffer, new_cap);
            if (!tmp) { free(buffer); return 1; }
            buffer = tmp;
            capacity = new_cap;
        }

        int ch = getchar();
        if (ch == EOF) {
            if (ferror(stdin)) { free(buffer); return 1; }
            break;
        }
        buffer[length++] = (unsigned char)ch;
    }

    /* shrink-to-fit optional */
    if (length == 0) {
        /* allow empty input; still return a valid pointer or NULL? choose NULL */
        free(buffer);
        *data = NULL;
        *size = 0;
        return 0;
    }

    unsigned char *tight = (unsigned char*)realloc(buffer, length);
    *data = tight ? tight : buffer;
    *size = length;
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* IMG_UTILS_WIN_H */
