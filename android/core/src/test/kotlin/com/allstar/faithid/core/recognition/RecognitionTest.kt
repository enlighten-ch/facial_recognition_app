package com.allstar.faithid.core.recognition

import com.allstar.faithid.core.geometry.GuideEllipse
import com.allstar.faithid.core.geometry.Point
import com.allstar.faithid.core.model.PersonMetadata
import com.allstar.faithid.core.recognition.CandidateStabilizer.Kind
import com.allstar.faithid.core.recognition.CandidateStabilizer.Observation
import com.allstar.faithid.core.testing.FakeFaces
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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

    private fun facing(d: HeadDirection) = Pose.Facing(d)

    @Test
    fun `classifies front, the four directions, in-between and too far, and mirrors yaw for the front camera`() {
        assertEquals(facing(HeadDirection.FRONT), HeadPose.classify(landmarks(60f, 75f), mirroredPreview = false))
        assertEquals(facing(HeadDirection.LEFT), HeadPose.classify(landmarks(52f, 75f), false))
        assertEquals(facing(HeadDirection.RIGHT), HeadPose.classify(landmarks(52f, 75f), true))
        assertEquals(facing(HeadDirection.RIGHT), HeadPose.classify(landmarks(68f, 75f), false))
        assertEquals(facing(HeadDirection.UP), HeadPose.classify(landmarks(60f, 65f), false))
        assertEquals(facing(HeadDirection.DOWN), HeadPose.classify(landmarks(60f, 85f), false))
        // A little turned or tilted: neither front nor a direction.
        assertEquals(Pose.Between, HeadPose.classify(landmarks(55f, 75f), false))
        assertEquals(Pose.Between, HeadPose.classify(landmarks(60f, 80f), false))
        // Nearly a profile, or the chin far up: landmarks are unreliable there.
        assertEquals(Pose.TooFar, HeadPose.classify(landmarks(30f, 75f), false))
        assertEquals(Pose.TooFar, HeadPose.classify(landmarks(60f, 54f), false))
    }

    @Test
    fun `measures yaw in eye distances and pitch between eyes and mouth`() {
        val angles = HeadPose.measure(landmarks(52f, 65f), false)!!
        assertEquals(-0.2f, angles.yaw, 1e-4f)
        assertEquals(0.3f, angles.pitch, 1e-4f)
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

    private fun frame(direction: HeadDirection?, size: Float? = null, identity: String = "me", sharpness: Float = 0f) =
        RegistrationFrame(faces.sample(identity), direction?.let { Pose.Facing(it) } ?: Pose.Between, size, sharpness)

    /** Offers the same pose until a sample is kept or refused; returns the last outcome. */
    private fun RegistrationSession.hold(direction: HeadDirection?, size: Float? = null, identity: String = "me"): RegistrationSession.Outcome {
        var outcome: RegistrationSession.Outcome
        do outcome = offer(frame(direction, size, identity)) while (outcome is RegistrationSession.Outcome.Holding)
        return outcome
    }

    private val accepted = RegistrationSession.Outcome.Accepted::class

    @Test
    fun `fifteen cells, front included, suggested by circle round`() {
        val session = RegistrationSession(RegistrationConfig())
        assertEquals(15, session.totalTarget)
        assertEquals(RegistrationCell(HeadDirection.FRONT, FaceCircle.NORMAL), session.suggestion)
        assertEquals(HeadDirection.entries, session.cells.take(5).map { it.direction })
        assertTrue(session.cells.take(5).all { it.circle == FaceCircle.NORMAL })
        assertTrue(session.cells.drop(5).take(5).all { it.circle == FaceCircle.BIG })
        assertEquals(12, RegistrationSession(RegistrationConfig(includeFront = false)).totalTarget)
    }

    @Test
    fun `fills whatever empty cell the pose and distance match, in any order`() {
        val session = RegistrationSession(RegistrationConfig())
        // The guide suggests front/normal, but a small, upward face fills up/small right away.
        val up = session.hold(HeadDirection.UP, 0.73f)
        assertEquals(RegistrationCell(HeadDirection.UP, FaceCircle.SMALL), (up as RegistrationSession.Outcome.Accepted).cell)
        assertEquals(RegistrationCell(HeadDirection.FRONT, FaceCircle.NORMAL), session.suggestion)
        listOf(1f, 1.18f, 0.73f).forEach { size ->
            HeadDirection.entries.forEach { d ->
                if (!session.has(RegistrationCell(d, session.circleFor(size)!!))) assertTrue(accepted.isInstance(session.hold(d, size)))
            }
        }
        assertTrue(session.isComplete)
        assertEquals(15, session.samples.size)
        assertEquals(RegistrationSession.Outcome.AlreadyComplete, session.offer(frame(HeadDirection.LEFT, 1f)))
        session.reset()
        assertEquals(0, session.totalCollected)
    }

    @Test
    fun `keeps a sample only after the hold, and the sharpest frame of it`() {
        val session = RegistrationSession(RegistrationConfig(holdFrames = 3))
        val blurry = frame(HeadDirection.FRONT, 1f, sharpness = 1f)
        val sharp = frame(HeadDirection.FRONT, 1f, sharpness = 9f)
        assertEquals(RegistrationSession.Outcome.Holding(RegistrationCell(HeadDirection.FRONT, FaceCircle.NORMAL), 1, 3), session.offer(blurry))
        assertTrue(session.offer(sharp) is RegistrationSession.Outcome.Holding)
        assertTrue(accepted.isInstance(session.offer(frame(HeadDirection.FRONT, 1f, sharpness = 2f))))
        assertTrue(session.samples.single() === sharp.embedding)
    }

    @Test
    fun `a different pose, no face or several faces start the hold again`() {
        val session = RegistrationSession(RegistrationConfig(holdFrames = 2))
        assertTrue(session.offer(frame(HeadDirection.FRONT, 1f)) is RegistrationSession.Outcome.Holding)
        assertTrue(session.offer(frame(null, 1f)) is RegistrationSession.Outcome.WrongDirection)
        assertTrue(session.offer(frame(HeadDirection.FRONT, 1f)) is RegistrationSession.Outcome.Holding)
        session.interrupt()
        assertTrue(session.offer(frame(HeadDirection.FRONT, 1f)) is RegistrationSession.Outcome.Holding)
        assertTrue(accepted.isInstance(session.offer(frame(HeadDirection.FRONT, 1f))))
    }

    @Test
    fun `size is judged against the circles as drawn`() {
        val session = RegistrationSession(RegistrationConfig())
        assertEquals(FaceCircle.NORMAL, session.circleFor(1.05f))
        assertEquals(FaceCircle.BIG, session.circleFor(1.1f))   // nearer the big circle than the normal one
        assertEquals(FaceCircle.SMALL, session.circleFor(0.66f))
        assertNull(session.circleFor(0.85f))
        assertNull(session.circleFor(1.5f))
        // Between the normal and the small circle: no cell; the guide asks for the suggested circle.
        val between = session.hold(HeadDirection.FRONT, 0.85f) as RegistrationSession.Outcome.WrongSize
        assertEquals(FaceCircle.NORMAL, between.expected)
        assertEquals(0.85f, between.ratio, 1e-4f)
        // Front/normal done: a normal-sized front face asks for the next open front circle.
        session.hold(HeadDirection.FRONT, 1f)
        assertEquals(FaceCircle.BIG, (session.hold(HeadDirection.FRONT, 1f) as RegistrationSession.Outcome.WrongSize).expected)
    }

    @Test
    fun `refuses a pose turned too far, in between, or already collected`() {
        val session = RegistrationSession(RegistrationConfig(includeFront = false, circleSteps = false, samplesPerDirection = 1))
        assertEquals(RegistrationSession.Outcome.TooTurned, session.offer(RegistrationFrame(faces.sample("me"), Pose.TooFar)))
        assertEquals(RegistrationSession.Outcome.WrongDirection(HeadDirection.LEFT, null), session.hold(null))
        // Front is not collected in this configuration.
        assertEquals(RegistrationSession.Outcome.WrongDirection(HeadDirection.LEFT, HeadDirection.FRONT), session.hold(HeadDirection.FRONT))
        assertTrue(accepted.isInstance(session.hold(HeadDirection.RIGHT)))
        assertEquals(RegistrationSession.Outcome.WrongDirection(HeadDirection.LEFT, HeadDirection.RIGHT), session.hold(HeadDirection.RIGHT))
    }

    @Test
    fun `with circle steps off or an unknown size the distance is not checked`() {
        val off = RegistrationSession(RegistrationConfig(circleSteps = false, samplesPerDirection = 2))
        assertEquals(10, off.totalTarget)
        assertTrue(accepted.isInstance(off.hold(HeadDirection.LEFT, 0.3f)))
        assertTrue(accepted.isInstance(off.hold(HeadDirection.LEFT, 2f)))
        assertEquals(2, off.collected(HeadDirection.LEFT))

        val unknown = RegistrationSession(RegistrationConfig())
        assertTrue(accepted.isInstance(unknown.hold(HeadDirection.LEFT)))
        assertEquals(1, unknown.collected(HeadDirection.LEFT))
    }

    @Test
    fun `rejects a different person and an identical frame`() {
        val session = RegistrationSession(RegistrationConfig(holdFrames = 1))
        val first = faces.sample("me")
        session.offer(RegistrationFrame(first, Pose.Facing(HeadDirection.LEFT)))
        assertTrue(session.offer(frame(HeadDirection.RIGHT, identity = "someone else")) is RegistrationSession.Outcome.TooDifferent)
        assertTrue(session.offer(RegistrationFrame(first.copyOf(), Pose.Facing(HeadDirection.RIGHT))) is RegistrationSession.Outcome.TooSimilar)
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
