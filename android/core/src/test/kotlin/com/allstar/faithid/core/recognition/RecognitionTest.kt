package com.allstar.faithid.core.recognition

import com.allstar.faithid.core.geometry.GuideEllipse
import com.allstar.faithid.core.geometry.Point
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.recognition.CandidateStabilizer.Kind
import com.allstar.faithid.core.recognition.CandidateStabilizer.Observation
import com.allstar.faithid.core.testing.FakeFaces
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

class FaceIndexAndPolicyTest {
    private val faces = FakeFaces(seed = 11, dim = 128)
    private val thresholds = RecognitionThresholds()

    private fun match(key: String, score: Float) = Match(key, PersonMetadata.of(key, "", ""), score)

    @Test
    fun `index ranks the right person first and is stable for ties`() {
        val people = listOf(faces.person("A", key = "a"), faces.person("B", key = "b"), faces.person("C", key = "c"))
        val index = FaceIndex.of(people)
        val ranking = index.rank(faces.sample("b"), topK = 3)
        assertEquals("b", ranking.first().personKey)
        assertTrue(ranking.first().score > 0.7f)
        assertTrue(ranking[1].score < 0.4f)
        assertEquals(3, ranking.size)
        assertEquals(1, index.rank(faces.sample("b"), topK = 0).size)
        assertTrue(FaceIndex.EMPTY.rank(faces.sample("b"), 3).isEmpty())
        assertEquals("B", index.metadataOf("b")?.name)
    }

    @Test
    fun `homonyms are separate entries`() {
        val first = faces.person("김철수", key = "k1")
        val second = faces.person("김철수", key = "k2")
        val index = FaceIndex.of(listOf(first, second))
        assertEquals(2, index.size)
        assertEquals("k2", index.best(faces.sample("k2"))?.personKey)
    }

    @Test
    fun `decision follows score and margin thresholds`() {
        assertTrue(RecognitionPolicy.decide(emptyList(), thresholds) is Decision.Unregistered)
        assertTrue(RecognitionPolicy.decide(listOf(match("a", 0.39f)), thresholds) is Decision.Unregistered)
        val confirmed = RecognitionPolicy.decide(listOf(match("a", 0.70f), match("b", 0.30f)), thresholds)
        assertTrue(confirmed is Decision.Confirmed)
        assertEquals(0.40f, (confirmed as Decision.Confirmed).margin, 1e-6f)
        val alone = RecognitionPolicy.decide(listOf(match("a", 0.45f)), thresholds)
        assertTrue(alone is Decision.Confirmed)
        val ambiguous = RecognitionPolicy.decide(listOf(match("a", 0.60f), match("b", 0.55f), match("c", 0.20f), match("d", 0.1f)), thresholds)
        assertTrue(ambiguous is Decision.Ambiguous)
        assertEquals(listOf("a", "b", "c"), (ambiguous as Decision.Ambiguous).candidates.map { it.personKey })
    }

    @Test
    fun `adaptive samples only between the recognition and adaptive limits`() {
        fun confirmed(score: Float) = Decision.Confirmed(match("a", score), 0.5f)
        assertTrue(RecognitionPolicy.isAdaptiveSample(confirmed(0.50f), thresholds))
        assertFalse(RecognitionPolicy.isAdaptiveSample(confirmed(0.70f), thresholds))
        assertFalse(RecognitionPolicy.isAdaptiveSample(Decision.Ambiguous(listOf(match("a", 0.5f))), thresholds))
    }
}

class CandidateStabilizerTest {
    private val ab = Observation(Kind.AMBIGUOUS, listOf("a", "b"))
    private val cd = Observation(Kind.AMBIGUOUS, listOf("c", "d"))

    @Test
    fun `first ambiguous group shows immediately and a new group needs three frames`() {
        val s = CandidateStabilizer(switchFrames = 3, graceFrames = 5)
        assertFalse(s.shouldHold(ab))
        s.onShown(ab)
        assertFalse(s.shouldHold(Observation(Kind.AMBIGUOUS, listOf("b", "a"))))
        assertTrue(s.shouldHold(cd))
        assertTrue(s.shouldHold(cd))
        assertFalse(s.shouldHold(cd))
    }

    @Test
    fun `a selected candidate survives short absences`() {
        val s = CandidateStabilizer(switchFrames = 1, graceFrames = 3)
        s.shouldHold(ab)
        s.onShown(ab)
        s.select("a")
        val other = Observation(Kind.UNREGISTERED)
        assertTrue(s.shouldHold(other))
        assertTrue(s.shouldHold(other))
        assertFalse(s.shouldHold(other))
        s.onShown(other)
        assertNull(s.selectedKey)
    }

    @Test
    fun `selection is kept while the new group still contains it`() {
        val s = CandidateStabilizer(switchFrames = 1, graceFrames = 2)
        s.shouldHold(ab)
        s.onShown(ab)
        s.select("b")
        assertTrue(s.shouldHold(Observation(Kind.AMBIGUOUS, listOf("b", "x"))))
        assertEquals("b", s.selectedKey)
    }

    @Test
    fun `confirmed selects the top person and reset clears everything`() {
        val s = CandidateStabilizer()
        val confirmed = Observation(Kind.CONFIRMED, listOf("z"))
        assertFalse(s.shouldHold(confirmed))
        s.onShown(confirmed)
        assertEquals("z", s.selectedKey)
        s.reset()
        assertNull(s.selectedKey)
    }
}

class HeadPoseAndGeometryTest {
    private fun landmarks(noseX: Float, noseY: Float) = FaceLandmarks(
        leftEye = Point(40f, 50f),
        rightEye = Point(80f, 50f),
        noseBase = Point(noseX, noseY),
        mouthLeft = Point(45f, 100f),
        mouthRight = Point(75f, 100f),
    )

    @Test
    fun `classifies yaw and pitch and mirrors yaw for the front camera`() {
        assertNull(HeadPose.classify(landmarks(60f, 75f), mirroredPreview = false))
        assertEquals(HeadDirection.LEFT, HeadPose.classify(landmarks(52f, 75f), false))
        assertEquals(HeadDirection.RIGHT, HeadPose.classify(landmarks(52f, 75f), true))
        assertEquals(HeadDirection.RIGHT, HeadPose.classify(landmarks(68f, 75f), false))
        assertEquals(HeadDirection.UP, HeadPose.classify(landmarks(60f, 65f), false))
        assertEquals(HeadDirection.DOWN, HeadPose.classify(landmarks(60f, 85f), false))
    }

    @Test
    fun `degenerate landmarks give no direction`() {
        val flat = FaceLandmarks(Point(0f, 0f), Point(10f, 0f), Point(5f, 0f), Point(0f, 0f), Point(10f, 0f))
        assertNull(HeadPose.classify(flat, false))
    }

    @Test
    fun `guide ellipse matches the original formula`() {
        val e = GuideEllipse.forFrame(480, 640, 0.22f, 0.30f, 0f)
        assertEquals(240f, e.centerX)
        assertEquals(320f, e.centerY)
        assertEquals(105f, e.radiusX)
        assertEquals(192f, e.radiusY)
        assertTrue(e.contains(240f, 320f))
        assertTrue(e.contains(240f + 104f, 320f))
        assertFalse(e.contains(240f + 106f, 320f))
        assertEquals(40f, GuideEllipse.forFrame(10, 10, 0.1f, 0.1f, 0f).radiusX)
    }

    @Test
    fun `alignment recovers a known similarity transform`() {
        val angle = 0.3
        val scale = 2.5f
        val a = (scale * cos(angle)).toFloat()
        val b = (scale * sin(angle)).toFloat()
        val truth = SimilarityTransform(a, b, 13f, -7f)
        val src = listOf(Point(10f, 20f), Point(40f, 18f), Point(25f, 35f), Point(14f, 50f), Point(37f, 52f))
        val dst = src.map { truth.apply(it) }
        val estimated = FaceAlignment.estimate(src, dst)
        assertEquals(truth.a, estimated.a, 1e-3f)
        assertEquals(truth.b, estimated.b, 1e-3f)
        assertEquals(truth.tx, estimated.tx, 1e-2f)
        assertEquals(truth.ty, estimated.ty, 1e-2f)
    }

    @Test
    fun `alignment maps landmarks near the ArcFace template regardless of eye labels`() {
        val template = FaceAlignment.ARCFACE_TEMPLATE
        val shifted = template.map { Point(it.x * 3f + 100f, it.y * 3f + 50f) }
        val landmarks = FaceLandmarks(
            leftEye = shifted[1], // ML Kit's "left eye" is the subject's left, i.e. image right
            rightEye = shifted[0],
            noseBase = shifted[2],
            mouthLeft = shifted[4],
            mouthRight = shifted[3],
        )
        val transform = FaceAlignment.transformFor(landmarks)
        val mapped = transform.apply(shifted[0])
        assertEquals(template[0].x, mapped.x, 1e-2f)
        assertEquals(template[0].y, mapped.y, 1e-2f)
        assertEquals(1f / 3f, transform.scale, 1e-3f)
    }
}

class RegistrationSessionTest {
    private val faces = FakeFaces(seed = 5, dim = 128)

    @Test
    fun `collects the configured samples per direction in order`() {
        val session = RegistrationSession(RegistrationConfig(samplesPerDirection = 2))
        assertEquals(HeadDirection.LEFT, session.nextDirection)
        val wrong = session.offer(HeadDirection.RIGHT, faces.sample("me"))
        assertTrue(wrong is RegistrationSession.Outcome.WrongDirection)
        HeadDirection.entries.forEach { direction ->
            repeat(2) { assertTrue(session.offer(direction, faces.sample("me")) is RegistrationSession.Outcome.Accepted) }
        }
        assertTrue(session.isComplete)
        assertEquals(8, session.samples.size)
        assertEquals(RegistrationSession.Outcome.AlreadyComplete, session.offer(HeadDirection.LEFT, faces.sample("me")))
        session.reset()
        assertEquals(0, session.totalCollected)
    }

    @Test
    fun `rejects a different person and an identical frame`() {
        val session = RegistrationSession(RegistrationConfig(samplesPerDirection = 3))
        val first = faces.sample("me")
        session.offer(HeadDirection.LEFT, first)
        assertTrue(session.offer(HeadDirection.LEFT, faces.sample("someone else")) is RegistrationSession.Outcome.TooDifferent)
        assertTrue(session.offer(HeadDirection.LEFT, first.copyOf()) is RegistrationSession.Outcome.TooSimilar)
    }

    @Test
    fun `circle steps ask for a closer and then a further face in each direction`() {
        val session = RegistrationSession(RegistrationConfig(samplesPerDirection = 3))
        assertEquals(FaceCircle.NORMAL, session.nextCircle)
        assertTrue(session.offer(HeadDirection.LEFT, faces.sample("me"), faceSize = 0.30f) is RegistrationSession.Outcome.Accepted)
        assertEquals(FaceCircle.BIG, session.nextCircle)
        val tooFar = session.offer(HeadDirection.LEFT, faces.sample("me"), faceSize = 0.31f)
        assertEquals(FaceCircle.BIG, (tooFar as RegistrationSession.Outcome.WrongSize).expected)
        assertTrue(session.offer(HeadDirection.LEFT, faces.sample("me"), faceSize = 0.36f) is RegistrationSession.Outcome.Accepted)
        assertEquals(FaceCircle.SMALL, session.nextCircle)
        assertTrue(session.offer(HeadDirection.LEFT, faces.sample("me"), faceSize = 0.29f) is RegistrationSession.Outcome.WrongSize)
        assertTrue(session.offer(HeadDirection.LEFT, faces.sample("me"), faceSize = 0.24f) is RegistrationSession.Outcome.Accepted)
        // The next direction starts again with the normal circle and its own baseline.
        assertEquals(HeadDirection.RIGHT, session.nextDirection)
        assertEquals(FaceCircle.NORMAL, session.nextCircle)
        assertTrue(session.offer(HeadDirection.RIGHT, faces.sample("me"), faceSize = 0.20f) is RegistrationSession.Outcome.Accepted)
        assertTrue(session.offer(HeadDirection.RIGHT, faces.sample("me"), faceSize = 0.23f) is RegistrationSession.Outcome.Accepted)
    }

    @Test
    fun `without a face size or with circle steps off the size is not checked`() {
        val unknown = RegistrationSession(RegistrationConfig(samplesPerDirection = 3))
        repeat(3) { assertTrue(unknown.offer(HeadDirection.LEFT, faces.sample("me")) is RegistrationSession.Outcome.Accepted) }

        val off = RegistrationSession(RegistrationConfig(samplesPerDirection = 3, circleSteps = false))
        repeat(3) { assertTrue(off.offer(HeadDirection.LEFT, faces.sample("me"), faceSize = 0.3f) is RegistrationSession.Outcome.Accepted) }
        assertEquals(FaceCircle.NORMAL, off.nextCircle)
    }

    @Test
    fun `detects that the face is already registered`() {
        val registered = faces.person("홍길동", key = "hong")
        val index = FaceIndex.of(listOf(registered, faces.person("다른사람", key = "other")))
        val session = RegistrationSession(RegistrationConfig(samplesPerDirection = 1))
        HeadDirection.entries.forEach { session.offer(it, faces.sample("hong")) }
        val duplicate = session.likelyDuplicateOf(index)
        assertNotNull(duplicate)
        assertEquals("hong", duplicate!!.personKey)

        val newcomer = RegistrationSession(RegistrationConfig(samplesPerDirection = 1))
        HeadDirection.entries.forEach { newcomer.offer(it, faces.sample("newcomer")) }
        assertNull(newcomer.likelyDuplicateOf(index))
    }

    @Test
    fun `adaptive sampler respects interval and per-session limit`() {
        val sampler = AdaptiveSampler(enabled = true, maxPerSession = 2, minIntervalFrames = 10)
        assertTrue(sampler.shouldTake(0))
        assertFalse(sampler.shouldTake(5))
        assertTrue(sampler.shouldTake(10))
        assertFalse(sampler.shouldTake(100))
        sampler.resetSession()
        assertTrue(sampler.shouldTake(101))
        assertFalse(AdaptiveSampler(enabled = false).shouldTake(0))
    }
}
