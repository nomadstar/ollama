package llm

import (
	"testing"
)

func TestTriAttentionVRAMReduction(t *testing.T) {
	tests := []struct {
		name       string
		pageBudget int
		ctx        int
		want       float64
	}{
		{"disabled budget", 0, 1024, 1.0},
		{"negative budget", -1, 1024, 1.0},
		{"zero ctx", 128, 0, 1.0},
		{"negative ctx", 128, -100, 1.0},
		{"budget equal to ctx", 1024, 1024, 1.0},
		{"budget larger than ctx", 2048, 1024, 1.0},
		{"valid budget", 512, 1024, 0.5},
		{"another valid budget", 256, 1024, 0.25},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := TriAttentionVRAMReduction(tt.pageBudget, tt.ctx)
			if got != tt.want {
				t.Errorf("TriAttentionVRAMReduction() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestPredictServerVRAMWithTriAttention(t *testing.T) {
	tests := []struct {
		name          string
		predictedVRAM uint64
		pageBudget    int
		ctx           int
		kvFraction    float64
		want          uint64
	}{
		{"no reduction", 1000, 0, 1024, 0.5, 1000},
		{"budget equals ctx", 1000, 1024, 1024, 0.5, 1000},
		{"budget larger than ctx", 1000, 2048, 1024, 0.5, 1000},
		{"50% reduction", 1000, 512, 1024, 0.5, 750},                   // 500 non-kv, 500 kv * 0.5 = 250 => 750
		{"25% budget with 40% kv fraction", 1000, 256, 1024, 0.4, 700}, // 600 non-kv, 400 kv * 0.25 = 100 => 700
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := PredictServerVRAMWithTriAttention(tt.predictedVRAM, tt.pageBudget, tt.ctx, tt.kvFraction)
			if got != tt.want {
				t.Errorf("PredictServerVRAMWithTriAttention() = %v, want %v", got, tt.want)
			}
		})
	}
}
