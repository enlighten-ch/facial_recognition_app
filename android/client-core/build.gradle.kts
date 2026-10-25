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
    api(project(":core"))
    api("com.squareup.okhttp3:okhttp:4.12.0")

    testImplementation(testFixtures(project(":core")))
    testImplementation(project(":server-core"))
    testImplementation("junit:junit:4.13.2")
}
