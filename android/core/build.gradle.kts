import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("org.jetbrains.kotlin.jvm")
    `java-test-fixtures`
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}

dependencies {
    // SQLite driver for JVM tests; Android uses the platform SQLite (see :android-support).
    testFixturesApi("org.xerial:sqlite-jdbc:3.46.1.3")
    testFixturesApi("junit:junit:4.13.2")
    testImplementation("junit:junit:4.13.2")
}
