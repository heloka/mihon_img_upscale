package eu.kanade.tachiyomi.util.waifu2x;

import android.os.ParcelFileDescriptor;
import eu.kanade.tachiyomi.util.waifu2x.IImageEnhancementNpuCallback;

interface IImageEnhancementNpuService {
    int getProcessId();
    void initialize(String modelPath, String nativeLibraryDir, int padding, boolean inputBgr, int requestId, IImageEnhancementNpuCallback callback);
    void process(in ParcelFileDescriptor input, in ParcelFileDescriptor output, int requestId, int imageId, int outputScale, IImageEnhancementNpuCallback callback);
    void abort(int imageId);
}
