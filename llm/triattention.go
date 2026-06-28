package llm

// TriAttentionVRAMReduction estimates VRAM savings from a page budget.
// pageBudget: number of KV pages to keep (tokens). ctx: full context length.
// Returns a multiplier in (0,1] to apply to the KV-cache portion of predictedVRAM.
func TriAttentionVRAMReduction(pageBudget, ctx int) float64 {
	if pageBudget <= 0 || ctx <= 0 || pageBudget >= ctx {
		return 1.0
	}
	return float64(pageBudget) / float64(ctx)
}

// PredictServerVRAMWithTriAttention returns the predicted VRAM with a TriAttention
// page budget applied. kvFraction is the fraction of predictedVRAM that is KV cache
// (use 0.5 as a safe default when unknown).
func PredictServerVRAMWithTriAttention(predictedVRAM uint64, pageBudget, ctx int, kvFraction float64) uint64 {
	if pageBudget <= 0 || pageBudget >= ctx {
		return predictedVRAM
	}
	reduction := TriAttentionVRAMReduction(pageBudget, ctx)
	kvPart := float64(predictedVRAM) * kvFraction
	nonKVPart := float64(predictedVRAM) * (1.0 - kvFraction)
	return uint64(nonKVPart + kvPart*reduction)
}
