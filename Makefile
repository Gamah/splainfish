# =============================================================================
# splainfish Makefile
#
# Targets
# -------
#   make setup         — clone/update latest Stockfish source, build it,
#                        export NNUE from system binary, install Python deps
#   make report        — analyse $(PGN) → $(OUTPUT)
#   make demo          — run on demo/immortal.pgn → demo/immortal.html
#   make clean         — remove build artefacts (keeps SF source + NNUE)
#   make distclean     — remove everything including SF source
#
# Variables (override on command line)
# ------------------------------------
#   PGN       path to PGN file          (default: demo/immortal.pgn)
#   OUTPUT    output HTML file          (default: report.html)
#   DEPTH     Stockfish search depth    (default: 18)
#   SF_ARGS   extra args for cli.py
#
# Examples
# --------
#   make setup
#   make report PGN=mygame.pgn OUTPUT=mygame.html
#   make demo
# =============================================================================

PGN    ?= demo/immortal.pgn
OUTPUT ?= report.html
DEPTH  ?= 18
SF_ARGS ?=

SF_DIR    := stockfish-src
SF_BIN    := $(SF_DIR)/src/stockfish
VENV      := .venv
PYTHON    := $(VENV)/bin/python
PIP       := $(VENV)/bin/pip

# System Stockfish binary (used for NNUE export if we can't download it)
SYS_SF    := $(shell which stockfish 2>/dev/null || echo /usr/games/stockfish)

# NNUE file: look for one next to either binary
NNUE_FILE := $(wildcard $(SF_DIR)/src/*.nnue) $(wildcard *.nnue) $(wildcard demo/*.nnue)
NNUE_FIRST := $(firstword $(NNUE_FILE))

# OS detection for build target
UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
  SF_ARCH := ARCH=apple-silicon
else
  SF_ARCH := ARCH=x86-64
endif
NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

.PHONY: all setup clone-sf build-sf export-nnue venv deps report demo clean distclean help

all: help

help:
	@echo ""
	@echo "  splainfish — NNUE activation probing for move explanation"
	@echo ""
	@echo "  make setup          Clone + build latest Stockfish, export NNUE, install deps"
	@echo "  make report         Analyse PGN=$(PGN) → $(OUTPUT)"
	@echo "  make demo           Analyse demo/immortal.pgn → demo/immortal.html"
	@echo "  make clean          Remove build artefacts"
	@echo "  make distclean      Remove everything including Stockfish source"
	@echo ""
	@echo "  Overrides: PGN=game.pgn  OUTPUT=out.html  DEPTH=18"
	@echo ""

# =============================================================================
# Setup
# =============================================================================
setup: clone-sf build-sf export-nnue venv deps
	@echo ""
	@echo "✓ Setup complete."
	@echo "  SF binary : $(SF_BIN)"
	@echo "  NNUE      : $(firstword $(wildcard $(SF_DIR)/src/*.nnue) unknown)"
	@echo "  Python env: $(VENV)"
	@echo ""
	@echo "  Run: make report PGN=your_game.pgn"

# Clone or update Stockfish from latest main
clone-sf:
	@if [ -d "$(SF_DIR)/.git" ]; then \
	  echo "→ Updating Stockfish source (git pull)..."; \
	  git -C $(SF_DIR) pull --ff-only; \
	else \
	  echo "→ Cloning latest Stockfish..."; \
	  git clone --depth=1 https://github.com/official-stockfish/Stockfish.git $(SF_DIR); \
	fi

# Build Stockfish with NNUE embedding disabled so it links without needing the .nnue
build-sf: clone-sf
	@echo "→ Building Stockfish ($(SF_ARCH), NNUE_EMBEDDING_OFF)..."
	$(MAKE) -C $(SF_DIR)/src -j$(NPROC) build $(SF_ARCH) COMP=gcc \
	  optimize=no EXTRACXXFLAGS="-DNNUE_EMBEDDING_OFF"
	@echo "✓ Built: $(SF_BIN)"

# Export the NNUE from the system Stockfish binary (works with apt-installed SF)
# Falls back gracefully if no system SF is found.
export-nnue:
	@NNUE_DEST="$(SF_DIR)/src/$$(echo 'uci' | $(SYS_SF) 2>/dev/null | grep 'EvalFile' | grep -o 'nn-[a-z0-9]*\.nnue')"; \
	if [ -n "$(NNUE_FIRST)" ]; then \
	  echo "✓ NNUE already present: $(NNUE_FIRST)"; \
	elif [ -x "$(SYS_SF)" ] && [ -n "$$(echo 'uci' | $(SYS_SF) 2>/dev/null | grep EvalFile)" ]; then \
	  echo "→ Exporting NNUE from system Stockfish ($(SYS_SF))..."; \
	  NNUE_NAME=$$(echo 'uci' | $(SYS_SF) 2>/dev/null | grep EvalFile | grep -o 'nn-[a-z0-9]*\.nnue'); \
	  printf 'uci\nexport_net $(SF_DIR)/src/%s\nquit\n' "$$NNUE_NAME" | $(SYS_SF) > /dev/null 2>&1; \
	  if [ -f "$(SF_DIR)/src/$$NNUE_NAME" ]; then \
	    echo "✓ Exported: $(SF_DIR)/src/$$NNUE_NAME"; \
	  else \
	    echo "⚠ Export failed. Place a .nnue file in $(SF_DIR)/src/ manually."; \
	  fi; \
	else \
	  echo "⚠ No system Stockfish found. Place a .nnue file in $(SF_DIR)/src/ manually."; \
	  echo "  Download from: https://github.com/official-stockfish/networks"; \
	fi

venv:
	@[ -d "$(VENV)" ] || python3 -m venv $(VENV)

deps: venv
	@echo "→ Installing Python dependencies..."
	@$(PIP) install --quiet --upgrade pip
	@$(PIP) install --quiet -r requirements.txt
	@echo "✓ Dependencies installed"

# =============================================================================
# Report generation
# =============================================================================

# Locate NNUE: prefer one in SF build dir, then anywhere in project
_nnue := $(firstword $(wildcard $(SF_DIR)/src/*.nnue) $(wildcard *.nnue) $(wildcard demo/*.nnue))

report: $(PYTHON) $(PGN)
	@if [ -z "$(_nnue)" ]; then echo "✗ No .nnue file found. Run: make setup"; exit 1; fi
	@if [ ! -f "$(SF_BIN)" ] && [ ! -x "$(SYS_SF)" ]; then \
	  echo "✗ No Stockfish binary. Run: make setup"; exit 1; fi
	@SF=$$([ -f "$(SF_BIN)" ] && echo "$(SF_BIN)" || echo "$(SYS_SF)"); \
	echo "→ Analysing $(PGN) at depth $(DEPTH)..."; \
	$(PYTHON) -m splainfish.cli \
	  --pgn "$(PGN)" \
	  --html "$(OUTPUT)" \
	  --stockfish "$$SF" \
	  --nnue "$(_nnue)" \
	  --depth $(DEPTH) \
	  $(SF_ARGS) \
	  --verbose; \
	echo "✓ Report: $(OUTPUT)"

# =============================================================================
# Demo
# =============================================================================
demo/immortal.pgn:
	@mkdir -p demo
	@printf '[Event "Casual game"]\n[Site "Vienna, AUL"]\n[Date "1851.??.??"]\n[Round "?"]\n[White "Anderssen, A"]\n[Black "Kieseritzky, L"]\n[Result "1-0"]\n\n1. e4 e5 2. f4 exf4 3. Bc4 Qh4+ 4. Kf1 b5 5. Bxb5 Nf6 6. Nf3 Qh6 7. d3 Nh5\n8. Nh4 Qg5 9. Nf5 c6 10. g4 Nf6 11. Rg1 cxb5 12. h4 Qg6 13. h5 Qg5 14. Qf3\nNg8 15. Bxf4 Qf6 16. Nc3 Bc5 17. Nd5 Qxb2 18. Bd6 Bxg1 19. e5 Qxa1+ 20. Ke2\nNa6 21. Nxg7+ Kd8 22. Qf6+ Nxf6 23. Be7# 1-0\n' > demo/immortal.pgn

demo: demo/immortal.pgn
	$(MAKE) report PGN=demo/immortal.pgn OUTPUT=demo/immortal.html
	@echo "✓ Demo → demo/immortal.html"
	@command -v xdg-open >/dev/null 2>&1 && xdg-open demo/immortal.html || \
	 command -v open     >/dev/null 2>&1 && open     demo/immortal.html || true

# =============================================================================
# Cleanup
# =============================================================================
clean:
	rm -rf __pycache__ splainfish/__pycache__ *.pyc
	rm -f report.html

distclean: clean
	rm -rf $(VENV) $(SF_DIR) demo
