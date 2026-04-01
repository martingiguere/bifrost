package complexity

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ComplexityInput is the normalized input for the analyzer.
// The caller is responsible for extracting text from request payloads.
type ComplexityInput struct {
	LastUserText   string   // last user message text
	PriorUserTexts []string // previous user message texts (up to 10)
	SystemText     string   // concatenated system/developer prompt text
}

// ComplexityContributions captures the weighted contribution of each lexical
// dimension to the last-message score before conversation blending/floors.
type ComplexityContributions struct {
	Code          float64
	Reasoning     float64
	Technical     float64
	SimplePenalty float64
	TokenCount    float64
}

// ComplexityResult holds the computed complexity scores and tier classification.
type ComplexityResult struct {
	// Weighted total score (0.0–1.0)
	Score float64
	// Computed tier: "SIMPLE", "MEDIUM", "COMPLEX", or "REASONING"
	Tier string

	// Individual dimension scores used in the weighted sum (0.0–1.0 each)
	CodePresence       float64
	ReasoningMarkers   float64
	TechnicalTerms     float64
	SimpleIndicators   float64
	TokenCount         float64
	ConversationCtx    float64
	SystemPromptSignal float64 // net weighted contribution from system lexical assist
	OutputComplexity   float64

	// Weighted contributions for the last message score.
	Contributions ComplexityContributions

	// Debug info — match counts per dimension (for logging, not exposed to CEL)
	CodeMatchCount      int
	ReasoningMatchCount int
	TechnicalMatchCount int
	SimpleMatchCount    int
	OutputMatchCount    int

	// Debug internals for eval/logging
	LastMessageScore    float64
	ConversationBlend   float64
	ReferentialFollowup bool
	SimpleWeightApplied float64
	OutputFloorMinScore float64
	OutputFloorApplied  bool
	WordCount           int
}

// TierBoundaries defines the score thresholds for tier classification.
type TierBoundaries struct {
	SimpleMedium     float64 `json:"simple_medium"`
	MediumComplex    float64 `json:"medium_complex"`
	ComplexReasoning float64 `json:"complex_reasoning"`
}

// DefaultTierBoundaries returns the default tier boundary thresholds.
func DefaultTierBoundaries() TierBoundaries {
	return TierBoundaries{
		SimpleMedium:     0.15,
		MediumComplex:    0.35,
		ComplexReasoning: 0.60,
	}
}

// Validate checks that tier boundaries satisfy the required ordering.
func (b *TierBoundaries) Validate() error {
	if !(0 < b.SimpleMedium &&
		b.SimpleMedium < b.MediumComplex &&
		b.MediumComplex < b.ComplexReasoning &&
		b.ComplexReasoning < 1) {
		return fmt.Errorf("tier boundaries must satisfy 0 < simple_medium (%.4f) < medium_complex (%.4f) < complex_reasoning (%.4f) < 1",
			b.SimpleMedium, b.MediumComplex, b.ComplexReasoning)
	}
	return nil
}

// EditableKeywordConfig is the user-facing subset of keyword lists that can be
// edited through the governance UI and config file. Every other keyword list
// the analyzer uses (weak reasoning, referential phrases, task-shift phrases,
// enum/comprehensiveness/elaboration/limiting markers) is an analyzer internal
// and always resolves from built-in defaults.
//
// ReasoningKeywords maps to the tier-override gate (strongReasoningKeywords),
// because that is what users actually want to control when they say "route
// these phrases to the reasoning model".
type EditableKeywordConfig struct {
	CodeKeywords      []string `json:"code_keywords"`
	ReasoningKeywords []string `json:"reasoning_keywords"`
	TechnicalKeywords []string `json:"technical_keywords"`
	SimpleKeywords    []string `json:"simple_keywords"`
}

// KeywordConfig is the full internal keyword configuration used by the
// compiled matcher. It is assembled from EditableKeywordConfig + defaults at
// analyzer-build time and is not part of the persisted or API surface.
type KeywordConfig struct {
	CodeKeywords              []string
	StrongReasoningKeywords   []string
	WeakReasoningKeywords     []string
	TechnicalKeywords         []string
	SimpleKeywords            []string
	EnumTriggers              []string
	ComprehensivenessMarkers  []string
	ElaborationMarkers        []string
	LimitingQualifiers        []string
	ReferentialPhrases        []string
	ReferentialReferenceWords []string
	ReferentialActionWords    []string
	TaskShiftPhrases          []string
}

// AnalyzerConfig is the full runtime configuration for the complexity analyzer.
// Persisted to the governance config store and exchanged over the management
// API in this shape.
type AnalyzerConfig struct {
	TierBoundaries TierBoundaries        `json:"tier_boundaries"`
	Keywords       EditableKeywordConfig `json:"keywords"`
}

// Validate checks that the analyzer config is internally consistent: tier
// boundaries are strictly ordered inside (0, 1) and every editable keyword
// list is non-empty after normalization. An empty list would wipe a scoring
// dimension, so it is rejected at the API boundary.
func (c *AnalyzerConfig) Validate() error {
	if c == nil {
		return nil
	}
	if err := c.TierBoundaries.Validate(); err != nil {
		return err
	}
	var missing []string
	if len(c.Keywords.CodeKeywords) == 0 {
		missing = append(missing, "code_keywords")
	}
	if len(c.Keywords.ReasoningKeywords) == 0 {
		missing = append(missing, "reasoning_keywords")
	}
	if len(c.Keywords.TechnicalKeywords) == 0 {
		missing = append(missing, "technical_keywords")
	}
	if len(c.Keywords.SimpleKeywords) == 0 {
		missing = append(missing, "simple_keywords")
	}
	if len(missing) > 0 {
		return fmt.Errorf("keyword lists must be non-empty: %s", strings.Join(missing, ", "))
	}
	return nil
}

// Normalized returns a canonical copy suitable for persistence and runtime use.
// Keyword entries are lowercased, trimmed, and deduplicated.
func (c *AnalyzerConfig) Normalized() AnalyzerConfig {
	if c == nil {
		return DefaultAnalyzerConfig()
	}

	return AnalyzerConfig{
		TierBoundaries: c.TierBoundaries,
		Keywords: EditableKeywordConfig{
			CodeKeywords:      normalizeKeywordList(c.Keywords.CodeKeywords),
			ReasoningKeywords: normalizeKeywordList(c.Keywords.ReasoningKeywords),
			TechnicalKeywords: normalizeKeywordList(c.Keywords.TechnicalKeywords),
			SimpleKeywords:    normalizeKeywordList(c.Keywords.SimpleKeywords),
		},
	}
}

// ValidateAndNormalize normalizes and validates an AnalyzerConfig in one step.
// If cfg is nil, returns default config. This is the canonical entrypoint that
// all callers should use instead of open-coding Normalized() + Validate().
func ValidateAndNormalize(cfg *AnalyzerConfig) (*AnalyzerConfig, error) {
	if cfg == nil {
		d := DefaultAnalyzerConfig()
		return &d, nil
	}
	normalized := cfg.Normalized()
	if err := normalized.Validate(); err != nil {
		return nil, err
	}
	return &normalized, nil
}

// DecodeAndValidate decodes raw JSON into an AnalyzerConfig, normalizes, and
// validates it. Returns nil with no error when data is nil or empty (missing
// config). This is the canonical entrypoint for callers reading from the
// config store's raw JSON representation.
func DecodeAndValidate(data []byte) (*AnalyzerConfig, error) {
	if len(data) == 0 {
		return nil, nil
	}
	var cfg AnalyzerConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal complexity analyzer config: %w", err)
	}
	normalized, err := ValidateAndNormalize(&cfg)
	if err != nil {
		return nil, fmt.Errorf("invalid complexity analyzer config: %w", err)
	}
	return normalized, nil
}

// MergeOntoDefaults overlays the editable keyword lists onto the built-in
// defaults and returns the full internal KeywordConfig used by the compiled
// matcher. Empty editable lists fall through to defaults as a defensive guard;
// Validate() should have already rejected them at the API boundary.
func (e EditableKeywordConfig) MergeOntoDefaults() KeywordConfig {
	kw := defaultFullKeywordConfig()
	if len(e.CodeKeywords) > 0 {
		kw.CodeKeywords = append([]string(nil), e.CodeKeywords...)
	}
	if len(e.ReasoningKeywords) > 0 {
		kw.StrongReasoningKeywords = append([]string(nil), e.ReasoningKeywords...)
	}
	if len(e.TechnicalKeywords) > 0 {
		kw.TechnicalKeywords = append([]string(nil), e.TechnicalKeywords...)
	}
	if len(e.SimpleKeywords) > 0 {
		kw.SimpleKeywords = append([]string(nil), e.SimpleKeywords...)
	}
	return kw
}

// DefaultEditableKeywordConfig returns the user-visible default keyword lists.
// ReasoningKeywords seeds from strongReasoningKeywords because that is the
// list users are actually editing when they say "reasoning keywords".
func DefaultEditableKeywordConfig() EditableKeywordConfig {
	return EditableKeywordConfig{
		CodeKeywords:      cloneStringSlice(codeKeywords),
		ReasoningKeywords: cloneStringSlice(strongReasoningKeywords),
		TechnicalKeywords: cloneStringSlice(technicalKeywords),
		SimpleKeywords:    cloneStringSlice(simpleKeywords),
	}
}

// DefaultAnalyzerConfig returns the default thresholds and user-visible
// keyword lists.
func DefaultAnalyzerConfig() AnalyzerConfig {
	return AnalyzerConfig{
		TierBoundaries: DefaultTierBoundaries(),
		Keywords:       DefaultEditableKeywordConfig(),
	}
}

// defaultFullKeywordConfig returns the complete internal keyword set, with all
// 13 lists populated from the package-level defaults. Used by the matcher and
// by MergeOntoDefaults as the base that editable lists overlay onto.
func defaultFullKeywordConfig() KeywordConfig {
	return KeywordConfig{
		CodeKeywords:              cloneStringSlice(codeKeywords),
		StrongReasoningKeywords:   cloneStringSlice(strongReasoningKeywords),
		WeakReasoningKeywords:     cloneStringSlice(weakReasoningKeywords),
		TechnicalKeywords:         cloneStringSlice(technicalKeywords),
		SimpleKeywords:            cloneStringSlice(simpleKeywords),
		EnumTriggers:              cloneStringSlice(enumTriggers),
		ComprehensivenessMarkers:  cloneStringSlice(comprehensivenessMarkers),
		ElaborationMarkers:        cloneStringSlice(elaborationMarkers),
		LimitingQualifiers:        cloneStringSlice(limitingQualifiers),
		ReferentialPhrases:        cloneStringSlice(referentialPhrases),
		ReferentialReferenceWords: cloneStringSlice(referentialReferenceWords),
		ReferentialActionWords:    cloneStringSlice(referentialActionWords),
		TaskShiftPhrases:          cloneStringSlice(taskShiftPhrases),
	}
}

func normalizeKeywordList(values []string) []string {
	if len(values) == 0 {
		return nil
	}

	seen := make(map[string]struct{}, len(values))
	out := make([]string, 0, len(values))
	for _, value := range values {
		normalized := strings.ToLower(strings.TrimSpace(value))
		if normalized == "" {
			continue
		}
		if _, ok := seen[normalized]; ok {
			continue
		}
		seen[normalized] = struct{}{}
		out = append(out, normalized)
	}
	return out
}

func cloneStringSlice(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	return append([]string(nil), values...)
}
