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
            abiFilters.addAll(listOf("arm64-v8a", "x86_64"))
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
    description = "Copies 16 KB page-aligned libc++_shared.so from NDK into build intermediates"
    doLast {
        val sdkDir = project.providers.gradleProperty("sdk.dir").orNull
            ?: System.getenv("ANDROID_HOME")
            ?: System.getenv("ANDROID_SDK_ROOT")
        val ndkDir = android.ndkPath ?: "$sdkDir/ndk/${android.ndkVersion}"
        val abiMap = mapOf(
            "arm64-v8a" to "aarch64-linux-android",
            "x86_64" to "x86_64-linux-android"
        )
        val targetDir = layout.buildDirectory.dir("intermediates/ndk_libcxx").get().asFile
        abiMap.forEach { (abi, sysrootArch) ->
            val srcFile = file("$ndkDir/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/$sysrootArch/libc++_shared.so")
            if (srcFile.exists()) {
                val destDir = file("$targetDir/$abi")
                destDir.mkdirs()
                srcFile.copyTo(file("$destDir/libc++_shared.so"), overwrite = true)
            }
        }
    }
}

tasks.matching { it.name.startsWith("merge") && it.name.endsWith("NativeLibs") }.configureEach {
    dependsOn(copyNdkLibCxxShared)
}