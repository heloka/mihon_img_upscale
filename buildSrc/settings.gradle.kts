pluginManagement {
    // Aliyun mirrors first: plugins.gradle.org is unreachable from some networks
    // (TLS handshake reset), which otherwise breaks buildSrc plugin resolution.
    repositories {
        maven(url = "https://maven.aliyun.com/repository/gradle-plugin")
        maven(url = "https://maven.aliyun.com/repository/public")
        maven(url = "https://maven.aliyun.com/repository/google")
        gradlePluginPortal()
        mavenCentral()
        google()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.PREFER_SETTINGS)
    repositories {
        maven(url = "https://maven.aliyun.com/repository/public")
        maven(url = "https://maven.aliyun.com/repository/google")
        maven(url = "https://maven.aliyun.com/repository/gradle-plugin")
        mavenCentral()
        google()
    }
    versionCatalogs {
        create("libs") {
            from(files("../gradle/libs.versions.toml"))
        }
        create("androidx") {
            from(files("../gradle/androidx.versions.toml"))
        }
        create("compose") {
            from(files("../gradle/compose.versions.toml"))
        }
        create("kotlinx") {
            from(files("../gradle/kotlinx.versions.toml"))
        }
    }
}

rootProject.name = "mihon-buildSrc"
