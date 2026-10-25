package com.allstar.faithid.core.recognition

/**
 * Keeps the candidate buttons on screen steady while the camera keeps producing slightly different
 * rankings frame after frame.
 *
 * - A different set of ambiguous candidates replaces the current one only after it has been seen for
 *   [switchFrames] consecutive analyses.
 * - A person the user already tapped stays selected through up to [graceFrames] analyses in which
 *   they are missing from the ranking (a blink or a turned head).
 */
class CandidateStabilizer(
    private val switchFrames: Int = 3,
    private val graceFrames: Int = 5,
) {
    enum class Kind { NO_FACE, UNREGISTERED, CONFIRMED, AMBIGUOUS }

    /** What one analysis produced: its kind and the ranked person keys (best first). */
    data class Observation(val kind: Kind, val personKeys: List<String> = emptyList())

    var selectedKey: String? = null
        private set

    private var activeGroup: String? = null
    private var pendingGroup: String? = null
    private var pendingFrames = 0
    private var missingFrames = 0

    /**
     * Returns true when the screen should keep showing the current state instead of switching to
     * [observation]. Call [onShown] after the screen does switch.
     */
    fun shouldHold(observation: Observation): Boolean {
        if (observation.kind == Kind.AMBIGUOUS) return holdAmbiguous(observation.personKeys)
        val selected = selectedKey ?: return false
        if (selected in observation.personKeys) {
            missingFrames = 0
            return false
        }
        missingFrames++
        return missingFrames < graceFrames.coerceAtLeast(1)
    }

    /** Updates the stabilizer after the screen switched to [observation]. */
    fun onShown(observation: Observation) {
        when (observation.kind) {
            Kind.AMBIGUOUS -> {
                if (selectedKey != null && selectedKey !in observation.personKeys) clearSelection()
            }
            Kind.CONFIRMED -> {
                resetGroup()
                selectedKey = observation.personKeys.firstOrNull()
                missingFrames = 0
            }
            Kind.NO_FACE, Kind.UNREGISTERED -> {
                resetGroup()
                clearSelection()
            }
        }
    }

    fun select(personKey: String) {
        selectedKey = personKey
        missingFrames = 0
    }

    fun reset() {
        resetGroup()
        clearSelection()
    }

    private fun holdAmbiguous(keys: List<String>): Boolean {
        val group = keys.sorted().joinToString("|")
        if (activeGroup == null || activeGroup == group) {
            activeGroup = group
            pendingGroup = null
            pendingFrames = 0
            if (selectedKey == null || selectedKey in keys) missingFrames = 0
            return false
        }
        val selected = selectedKey
        if (selected != null) {
            if (selected in keys) {
                missingFrames = 0
                return true
            }
            missingFrames++
            if (missingFrames < graceFrames.coerceAtLeast(1)) return true
        }
        if (pendingGroup == group) pendingFrames++ else {
            pendingGroup = group
            pendingFrames = 1
        }
        if (pendingFrames < switchFrames.coerceAtLeast(1)) return true
        activeGroup = group
        pendingGroup = null
        pendingFrames = 0
        missingFrames = 0
        return false
    }

    private fun resetGroup() {
        activeGroup = null
        pendingGroup = null
        pendingFrames = 0
        missingFrames = 0
    }

    private fun clearSelection() {
        selectedKey = null
        missingFrames = 0
    }
}
