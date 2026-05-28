package sq.rogue.telemetry_bridge

import android.content.Context
import io.flutter.app.FlutterApplication
import com.secneo.sdk.Helper

class MainApplication : FlutterApplication() {
    override fun attachBaseContext(base: Context?) {
        super.attachBaseContext(base)
        // Crucial Secneo decryption call to unpack DJI SDK classes into the JVM ClassLoader at runtime
        Helper.install(this)
    }
}
