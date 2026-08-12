import java.io.File
import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.pv.androidfacefusion"
    compileSdk {
        version = release(36)
    }

    defaultConfig {
        applicationId = "com.pv.androidfacefusion"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        
        ndk {
            abiFilters.addAll(listOf("armeabi-v7a", "arm64-v8a", "x86", "x86_64"))
        }
    }

    ndkVersion = "28.2.13676358"

    val ndkLibCxxDir = layout.buildDirectory.dir("intermediates/ndk_libcxx")
    sourceSets.getByName("main") {
        jniLibs.directories.add(ndkLibCxxDir.get().asFile.path)
    }

    packaging {
        jniLibs {
            pickFirsts.add("**/libc++_shared.so")
            useLegacyPackaging = false
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    
    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.activity)
    implementation(libs.constraintlayout)
    
    // ONNX Runtime for running models
    implementation(libs.onnxruntime.android)
    
    // Glide for image loading (from URL and local)
    implementation(libs.glide)
    
    // OpenCV for image processing
    implementation(libs.opencv)
    
    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
}

val copyNdkLibCxxShared = tasks.register("copyNdkLibCxxShared") {
    description = "Copies 16 KB page-aligned libc++_shared.so from NDK into build intermediates across all host OS platforms"
    doLast {
        // 1. Resolve Android SDK directory reliably (reading local.properties first, then env vars)
        val sdkDir: File = run {
            val localPropsFile = project.rootDir.resolve("local.properties")
            if (localPropsFile.exists()) {
                val props = Properties()
                localPropsFile.inputStream().use { stream -> props.load(stream) }
                val path = props.getProperty("sdk.dir")
                if (!path.isNullOrBlank()) {
                    val f = file(path)
                    if (f.exists()) return@run f
                }
            }
            val envSdk = System.getenv("ANDROID_HOME") ?: System.getenv("ANDROID_SDK_ROOT")
            if (!envSdk.isNullOrBlank()) {
                val f = file(envSdk)
                if (f.exists()) return@run f
            }
            throw GradleException("Could not resolve Android SDK directory from local.properties or environment variables (ANDROID_HOME/ANDROID_SDK_ROOT).")
        }

        // 2. Resolve NDK directory
        val ndkDir: File = run {
            val configuredNdk = android.ndkPath
            if (!configuredNdk.isNullOrBlank()) {
                val f = file(configuredNdk)
                if (f.exists()) return@run f
            }
            val version = android.ndkVersion
            if (!version.isNullOrBlank()) {
                val f = sdkDir.resolve("ndk/$version")
                if (f.exists()) return@run f
            }
            val bundle = sdkDir.resolve("ndk-bundle")
            if (bundle.exists()) return@run bundle
            throw GradleException("Could not resolve NDK directory in SDK path ${sdkDir.absolutePath} for NDK version ${android.ndkVersion}.")
        }

        // 3. Resolve host prebuilt directory dynamically for cross-platform support (Linux, macOS, Windows)
        val prebuiltParent = ndkDir.resolve("toolchains/llvm/prebuilt")
        if (!prebuiltParent.exists() || !prebuiltParent.isDirectory) {
            throw GradleException("NDK LLVM prebuilt directory not found at ${prebuiltParent.absolutePath}.")
        }
        val hostDirs = prebuiltParent.listFiles { f -> f.isDirectory }
        if (hostDirs.isNullOrEmpty()) {
            throw GradleException("No host prebuilt directory found in ${prebuiltParent.absolutePath}.")
        }
        val hostPrebuiltDir = hostDirs.first()

        // 4. Map ABIs to sysroot library directories
        val abiMap = mapOf(
            "arm64-v8a" to "aarch64-linux-android",
            "x86_64" to "x86_64-linux-android",
            "armeabi-v7a" to "arm-linux-androideabi",
            "x86" to "i686-linux-android"
        )

        val targetDir = layout.buildDirectory.dir("intermediates/ndk_libcxx").get().asFile

        // 5. Copy libraries with explicit failure if source is missing
        abiMap.forEach { (abi, sysrootArch) ->
            val srcFile = hostPrebuiltDir.resolve("sysroot/usr/lib/$sysrootArch/libc++_shared.so")
            if (!srcFile.exists()) {
                throw GradleException("Required 16 KB compliant libc++_shared.so not found at ${srcFile.absolutePath} for ABI $abi! Cannot guarantee 16 KB page-size compliance.")
            }
            val destDir = targetDir.resolve(abi)
            destDir.mkdirs()
            val destFile = destDir.resolve("libc++_shared.so")
            srcFile.copyTo(destFile, overwrite = true)
            if (!destFile.exists()) {
                throw GradleException("Failed to copy libc++_shared.so to ${destFile.absolutePath}.")
            }
        }
    }
}

tasks.matching { it.name.startsWith("merge") && it.name.endsWith("NativeLibs") }.configureEach {
    dependsOn(copyNdkLibCxxShared)
}