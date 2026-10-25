pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "0.10.0"
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "faith-id"

// Pure Kotlin (JVM) modules: business logic, unit-testable without Android.
include(":core")
include(":server-core")
include(":client-core")
include(":integration-tests")

// Android modules.
include(":android-support")
include(":kiosk-app")
include(":server-app")
