import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("org.jetbrains.kotlin.jvm")
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}

dependencies {
    testImplementation(project(":server-core"))
    testImplementation(project(":client-core"))
    testImplementation(testFixtures(project(":core")))
    testImplementation("junit:junit:4.13.2")
}

tasks.test {
    // Multi-kiosk scenarios run for tens of seconds each; show progress in the console.
    testLogging {
        events("passed", "failed", "skipped")
        showStandardStreams = true
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
    maxHeapSize = "2g"
    // A real Android server can be targeted with -Pfaith.it.serverUrl=http://host:18080
    project.findProperty("faith.it.serverUrl")?.let { systemProperty("faith.it.serverUrl", it) }
    project.findProperty("faith.it.seed")?.let { systemProperty("faith.it.seed", it) }
}
