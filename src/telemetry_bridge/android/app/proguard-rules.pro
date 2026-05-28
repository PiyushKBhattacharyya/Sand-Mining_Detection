# Keep DJI SDK classes from being stripped or obfuscated
-keep class dji.** { *; }
-keep interface dji.** { *; }
-keep class com.dji.** { *; }
-keep class dji.sdk.sdkmanager.DJISDKManager { *; }

# Keep standard AndroidX and support library layouts
-keep class androidx.constraintlayout.** { *; }
-keep class androidx.appcompat.** { *; }
-dontwarn dji.**
