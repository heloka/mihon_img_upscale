package eu.kanade.tachiyomi.util.waifu2x;

interface IImageEnhancementNpuCallback {
    void onProgress(int requestId, int imageId, int percent);
    void onResult(int requestId, boolean success, int backend, int width, int height, String error);
}
