#ifndef IMG_UTILS_H
#define IMG_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32)
#include <wchar.h>
#define img_fopen_utf8(path, mode) _wfopen((const wchar_t*)path, (const wchar_t*)mode)
#endif

/* ---------- FILE LOAD ---------- */
int img_load(const void *path, unsigned char **img, size_t *img_size)
{
    FILE *file;

#if defined(_WIN32)
    /* If path is wide (UTF‑16), use Windows native API */
    if (((uintptr_t)path & 1) == 0) {
        file = img_fopen_utf8(path, L"rb");
    } else {
        file = fopen((const char*)path, "rb");
    }
#else
    file = fopen((const char*)path, "rb");
#endif

    if (!file) return 1;

#if defined(_WIN32)
    __int64 size = _ftelli64(file);
#else
    long size = ftell(file);
#endif

    fseek(file, 0, SEEK_END);
#if defined(_WIN32)
    size = _ftelli64(file);
#else
    size = ftell(file);
#endif
    fseek(file, 0, SEEK_SET);

    *img_size = (size_t)size;

    *img = (unsigned char*) malloc(*img_size);
    if (!*img) {
        fclose(file);
        return 2;
    }

    if (fread(*img, 1, *img_size, file) != *img_size) {
        free(*img);
        fclose(file);
        return 3;
    }

    fclose(file);
    return 0;
}

/* ---------- FILE SAVE ---------- */
int img_save(const void *path, const unsigned char *img, size_t img_size)
{
    FILE *file;

#if defined(_WIN32)
    if (((uintptr_t)path & 1) == 0) {
        file = img_fopen_utf8(path, L"wb");
    } else {
        file = fopen((const char*)path, "wb");
    }
#else
    file = fopen((const char*)path, "wb");
#endif

    if (!file) return 1;

    if (fwrite(img, 1, img_size, file) != img_size) {
        fclose(file);
        return 2;
    }

    fclose(file);
    return 0;
}

/* ---------- DESTROY ---------- */
int img_destroy(void *img)
{
    if (!img) return 1;
    free(img);
    return 0;
}

/* ---------- LOAD FROM STDIN ---------- */
int img_load_stdin(unsigned char **data, size_t *size)
{
    size_t cap = 4096, len = 0;
    unsigned char *buf = malloc(cap);
    if (!buf) return 1;

    int c;
    while ((c = getchar()) != EOF) {
        if (len >= cap) {
            cap <<= 1;
            unsigned char *tmp = realloc(buf, cap);
            if (!tmp) {
                free(buf);
                return 1;
            }
            buf = tmp;
        }
        buf[len++] = (unsigned char)c;
    }

    *data = buf;
    *size = len;
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif
