package sq.rogue.telemetry_bridge

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import androidx.annotation.NonNull
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.util.ArrayList

// Import DJI SDK classes
import dji.common.error.DJIError
import dji.common.error.DJISDKError
import dji.sdk.base.BaseProduct
import dji.sdk.products.Aircraft
import dji.sdk.sdkmanager.DJISDKManager
import dji.common.flightcontroller.FlightControllerState
import dji.common.battery.BatteryState

class MainActivity : FlutterActivity() {
    private val CHANNEL = "sq.rogue.telemetry_bridge/dji"
    private var methodChannel: MethodChannel? = null
    private val handler = Handler(Looper.getMainLooper())

    private val REQUEST_PERMISSION_CODE = 12345
    private val REQUIRED_PERMISSION_LIST = arrayOf(
        Manifest.permission.WRITE_EXTERNAL_STORAGE,
        Manifest.permission.READ_PHONE_STATE,
        Manifest.permission.ACCESS_FINE_LOCATION,
        Manifest.permission.ACCESS_COARSE_LOCATION
    )

    override fun configureFlutterEngine(@NonNull flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        methodChannel = MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)

        methodChannel?.setMethodCallHandler { call, result ->
            when (call.method) {
                "startDJISDK" -> {
                    checkAndRequestPermissions()
                    result.success("Initialization and permission checks started.")
                }
                "getSDKStatus" -> {
                    val registered = DJISDKManager.getInstance().hasSDKRegistered()
                    result.success(registered)
                }
                else -> {
                    result.notImplemented()
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Automatically trigger permission check on app launch
        handler.postDelayed({ checkAndRequestPermissions() }, 1000)
    }

    private fun checkAndRequestPermissions() {
        val missingPermissions = ArrayList<String>()
        for (permission in REQUIRED_PERMISSION_LIST) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                missingPermissions.add(permission)
            }
        }

        if (missingPermissions.isNotEmpty()) {
            sendConsoleLog("[SDK] Requesting required runtime permissions...")
            ActivityCompat.requestPermissions(
                this,
                missingPermissions.toTypedArray(),
                REQUEST_PERMISSION_CODE
            )
        } else {
            registerDJISDK()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_PERMISSION_CODE) {
            var allGranted = true
            for (result in grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    allGranted = false
                    break
                }
            }
            if (allGranted) {
                sendConsoleLog("[SDK] All permissions granted! Unlocking registration.")
                registerDJISDK()
            } else {
                sendConsoleLog("[SDK ERROR] Missing required permissions. DJI SDK registration aborted.")
            }
        }
    }

    private fun registerDJISDK() {
        sendConsoleLog("[SDK] Starting DJI SDK registration...")
        
        DJISDKManager.getInstance().registerApp(applicationContext, object : DJISDKManager.SDKManagerCallback {
            override fun onRegister(djiError: DJIError?) {
                if (djiError == DJISDKError.REGISTRATION_SUCCESS) {
                    sendConsoleLog("[SDK] DJI SDK App Key registered successfully!")
                    handler.post {
                        methodChannel?.invokeMethod("onSDKStatusUpdate", mapOf("status" to "REGISTERED"))
                    }
                    startConnectionListener()
                } else {
                    sendConsoleLog("[SDK] SDK Registration Failed: ${djiError?.description}")
                    handler.post {
                        methodChannel?.invokeMethod("onSDKStatusUpdate", mapOf(
                            "status" to "FAILED",
                            "error" to (djiError?.description ?: "Unknown Error")
                        ))
                    }
                }
            }

            override fun onProductDisconnect() {
                sendConsoleLog("[DJI] Drone Disconnected.")
                handler.post {
                    methodChannel?.invokeMethod("onDJIConnectionUpdate", false)
                }
            }

            override fun onProductConnect(product: BaseProduct?) {
                sendConsoleLog("[DJI] Drone Connected: ${product?.model?.displayName}")
                handler.post {
                    methodChannel?.invokeMethod("onDJIConnectionUpdate", true)
                }
                setupTelemetryListeners(product)
            }

            override fun onProductChanged(product: BaseProduct?) {
                sendConsoleLog("[DJI] Drone product changed.")
            }

            override fun onComponentChange(key: BaseProduct.ComponentKey?, oldComponent: dji.sdk.base.BaseComponent?, newComponent: dji.sdk.base.BaseComponent?) {
                sendConsoleLog("[DJI] Accessory component changed: ${key?.name}")
            }

            override fun onInitProcess(p0: dji.sdk.sdkmanager.DJISDKInitEvent?, p1: Int) {
                // DJI SDK init process log
            }

            override fun onDatabaseDownloadProgress(p0: Long, p1: Long) {
                // DJI SDK database download progress log
            }
        })
    }

    private fun startConnectionListener() {
        DJISDKManager.getInstance().startConnectionToProduct()
    }

    private fun setupTelemetryListeners(product: BaseProduct?) {
        if (product == null) return

        val aircraft = product as? Aircraft ?: return
        val flightController = aircraft.flightController

        if (flightController != null) {
            sendConsoleLog("[SDK] Binding to flight controller state callbacks...")
            flightController.setStateCallback { state: FlightControllerState ->
                val lat = state.aircraftLocation.latitude
                val lon = state.aircraftLocation.longitude
                val alt = state.aircraftLocation.altitude.toDouble()
                
                val speed = Math.sqrt(
                    Math.pow(state.velocityX.toDouble(), 2.0) +
                    Math.pow(state.velocityY.toDouble(), 2.0)
                )

                val telemetryData = mapOf(
                    "lat" to lat,
                    "lon" to lon,
                    "altitude" to alt,
                    "speed" to speed
                )

                handler.post {
                    methodChannel?.invokeMethod("onTelemetryUpdate", telemetryData)
                }
            }
        }

        val battery = aircraft.battery
        if (battery != null) {
            sendConsoleLog("[SDK] Binding to battery hardware callbacks...")
            battery.setStateCallback { state: BatteryState ->
                val batPercent = state.chargeRemainingInPercent
                handler.post {
                    methodChannel?.invokeMethod("onBatteryUpdate", batPercent)
                }
            }
        }
    }

    private fun sendConsoleLog(message: String) {
        handler.post {
            methodChannel?.invokeMethod("onConsoleLog", message)
        }
    }
}
