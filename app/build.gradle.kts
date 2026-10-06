import java.io.File
import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.pv.androidfacefusion"
    compileSdk {
        version = release(37)
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
        // Read local.properties if available
        val localPropsFile = project.rootDir.resolve("local.properties")
        val localProps = if (localPropsFile.exists()) {
            val props = Properties()
            localPropsFile.inputStream().use { stream -> props.load(stream) }
            props
        } else null

        // 1. Resolve Android SDK directory reliably (reading local.properties first, then env vars)
        val sdkDir: File = run {
            val localSdk = localProps?.getProperty("sdk.dir")
            if (!localSdk.isNullOrBlank()) {
                val f = file(localSdk)
                if (f.exists()) return@run f
            }
            val envSdk = System.getenv("ANDROID_HOME") ?: System.getenv("ANDROID_SDK_ROOT")
            if (!envSdk.isNullOrBlank()) {
                val f = file(envSdk)
                if (f.exists()) return@run f
            }
            throw GradleException("Could not resolve Android SDK directory from local.properties or environment variables (ANDROID_HOME/ANDROID_SDK_ROOT).")
        }

        // 2. Resolve NDK directory robustly across CI runners and local environments
        val ndkDir: File = run {
            // Priority 1: ndk.dir in local.properties
            val localNdk = localProps?.getProperty("ndk.dir")
            if (!localNdk.isNullOrBlank()) {
                val f = file(localNdk)
                if (f.exists()) return@run f
            }

            // Priority 2: Configured android.ndkPath
            val configuredNdk = android.ndkPath
            if (!configuredNdk.isNullOrBlank()) {
                val f = file(configuredNdk)
                if (f.exists()) return@run f
            }

            // Priority 3: Configured android.ndkVersion if present in $sdkDir/ndk/$version
            val version = android.ndkVersion
            if (!version.isNullOrBlank()) {
                val f = sdkDir.resolve("ndk/$version")
                if (f.exists()) return@run f
            }

            // Priority 4: Standard NDK environment variables (pre-set on GitHub Actions and CI runners)
            listOf("ANDROID_NDK_LATEST_HOME", "ANDROID_NDK_HOME", "ANDROID_NDK_ROOT", "ANDROID_NDK")
                .mapNotNull { System.getenv(it) }
                .filter { it.isNotBlank() }
                .map { file(it) }
                .firstOrNull { it.exists() }
                ?.let { return@run it }

            // Priority 5: Any installed NDK version in $sdkDir/ndk/ (pick highest version)
            val ndkParent = sdkDir.resolve("ndk")
            if (ndkParent.exists() && ndkParent.isDirectory) {
                val installedNdks = ndkParent.listFiles { f -> f.isDirectory && !f.name.startsWith(".") }
                if (!installedNdks.isNullOrEmpty()) {
                    val latestNdk = installedNdks.maxByOrNull { it.name }
                    if (latestNdk != null && latestNdk.exists()) {
                        return@run latestNdk
                    }
                }
            }

            // Priority 6: Legacy ndk-bundle fallback
            val bundle = sdkDir.resolve("ndk-bundle")
            if (bundle.exists()) return@run bundle

            throw GradleException(
                "Could not resolve Android NDK directory in SDK path ${sdkDir.absolutePath} " +
                "or from environment variables (ANDROID_NDK_LATEST_HOME, ANDROID_NDK_HOME, ANDROID_NDK_ROOT). " +
                "Please install an NDK (r27 or newer recommended for 16 KB page-size compliance) or set ANDROID_NDK_HOME."
            )
        }

        project.logger.lifecycle("Using Android NDK at: ${ndkDir.absolutePath}")

        // 3. Resolve host prebuilt directory dynamically for cross-platform support (Linux, macOS, Windows)
        val prebuiltParent = ndkDir.resolve("toolchains/llvm/prebuilt")
        if (!prebuiltParent.exists() || !prebuiltParent.isDirectory) {
            throw GradleException("NDK LLVM prebuilt directory not found at ${prebuiltParent.absolutePath}.")
        }
        val hostDirs = prebuiltParent.listFiles { f -> f.isDirectory && !f.name.startsWith(".") }
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