SHELL := /bin/bash
.PHONY: install

GREEN  := \033[0;32m
CYAN   := \033[0;36m
DIM    := \033[2m
RED    := \033[0;31m
BOLD   := \033[1m
NC     := \033[0m

define run_with_spinner
	$(2) > /dev/null 2>&1 & \
	pid=$$!; \
	spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'; \
	i=0; \
	start=$$(date +%s); \
	while kill -0 $$pid 2>/dev/null; do \
		i=$$(( (i+1) % 10 )); \
		elapsed=$$(( $$(date +%s) - start )); \
		printf "\r\033[K$(CYAN)  $${spin:$$i:1} $(1)$(DIM) ($${elapsed}s)$(NC)"; \
		sleep 0.1; \
	done; \
	wait $$pid; \
	elapsed=$$(( $$(date +%s) - start )); \
	if [ $$? -eq 0 ]; then \
		printf "\r\033[K$(GREEN)  ✓ $(1)$(DIM) ($${elapsed}s)$(NC)\n"; \
	else \
		printf "\r\033[K$(RED)  ✗ $(1)$(DIM) ($${elapsed}s)$(NC)\n"; \
		exit 1; \
	fi
endef

install:
	@if ! command -v uv >/dev/null 2>&1; then \
		$(call run_with_spinner,Installing uv,curl -LsSf https://astral.sh/uv/install.sh | sh); \
	fi
	@$(call run_with_spinner,Installing dependencies,uv sync --upgrade)
	@$(call run_with_spinner,Setting up pre-commit hooks,uv run pre-commit install)
	@printf "$(BOLD)$(GREEN)  ✓ Done! Environment ready.$(NC)\n"
	@printf "\n$(DIM)  Next steps:$(NC)\n"
	@printf "$(DIM)    source .venv/bin/activate$(NC)\n"
	@printf "$(DIM)    python src/mlp/train.py$(NC)\n"
	@printf "$(DIM)    python src/cnn/train.py$(NC)\n"
