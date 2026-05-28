plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "sq.rogue.telemetry_bridge"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_17.toString()
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "sq.rogue.telemetry_bridge"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = flutter.minSdkVersion // DJI Mobile SDK requires API level 21 or higher
        targetSdk = 33 // Target Android 13 (API 33) to bypass Android 14+ strict dynamic broadcast receiver checks
        versionCode = flutter.versionCode
        versionName = flutter.versionName
        
        // DJI SDK is extremely large; multidex is mandatory to avoid class loading limitations
        multiDexEnabled = true

        ndk {
            abiFilters.addAll(setOf("armeabi-v7a", "arm64-v8a"))
        }
    }

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
        debug {
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    packaging {
        jniLibs {
            keepDebugSymbols.add("**/*.so")
        }
        resources {
            excludes.add("META-INF/rxjava.properties")
            excludes.add("assets/location_map_gps_locked.png")
            excludes.add("assets/location_map_gps_3d.png")
        }
    }
}

dependencies {
    implementation("com.dji:dji-sdk:4.16.2")
    compileOnly("com.dji:dji-sdk-provided:4.16.2")

    // Required AndroidX libraries to resolve DJI SDK XML layout linking (ConstraintLayout & AppCompat)
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("com.google.android.material:material:1.9.0")
}

flutter {
    source = "../.."
}
