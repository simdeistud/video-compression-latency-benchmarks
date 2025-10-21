#ifndef IMG_UTILS_H
#define IMG_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>

/*
 * Loads an image from the specified file path.
 * Parameters:
 *   path      - path to the image file
 *   img       - pointer to the buffer that will hold the image data
 *   img_size  - pointer to the variable that will hold the size of the image
 * Returns:
 *   0 on success, non-zero on failure
 */
int img_load(const char path[], unsigned char** img, size_t* img_size) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        return 1; // Failed to open file
    }
    
    fseek(file, 0, SEEK_END);
    *img_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    *img = (unsigned char*) malloc(*img_size);
    if (!*img) {
        fclose(file);
        return 2; // Memory allocation failed
    }

    if (fread(*img, 1, *img_size, file) != *img_size) {
        free(*img);
        fclose(file);
        return 3; // Failed to read file
    }

    fclose(file);
    return 0; // Success
}

/*
 * Saves an image to the specified file path.
 * Parameters:
 *   path      - path to the output image file
 *   img       - buffer containing the image data
 *   img_size  - size of the image data
 * Returns:
 *   0 on success, non-zero on failure
 */
int img_save(char path[], unsigned char** img, size_t img_size) {
    FILE* file = fopen(path, "wb");
    if (!file) {
        return 1; // Failed to open file
    }

    if (fwrite(*img, 1, img_size, file) != img_size) {
        fclose(file);
        return 2; // Failed to write file
    }

    fclose(file);
    return 0; // Success
}

/*
 * Destroys a previously loaded image.
 * Parameters:
 *   img       - pointer to the buffer that holds the image data
 * Returns:
 *   0 on success, non-zero on failure
 */
int img_destroy(unsigned char* img) {
    if (!img) {
        return 1; // NULL pointer
    }
    free(img);
    return 0;
}

int img_load_stdin(unsigned char **data, size_t *size)
{
  
size_t capacity = 4096, length = 0;
    unsigned char *buffer = (unsigned char *) malloc(capacity);
    if (!buffer) return 1;

    int byte;
    while ((byte = getchar()) != EOF) {
        if (length >= capacity) {
            capacity *= 2;
            unsigned char *new_buffer = (unsigned char *) realloc(buffer, capacity);
            if (!new_buffer) {
                free(buffer);
                return 1;
            }
            buffer = new_buffer;
        }
        buffer[length++] = (unsigned char)byte;
    }

    *data = buffer;
    *size = length;
    return 0;


}

#ifdef __cplusplus
}
#endif

#endif // IMG_UTILS_H
