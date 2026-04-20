SHELL := /bin/bash
.PHONY: install run-mlp run-cnn clean help
.DEFAULT_GOAL := help

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

help:
	@printf "$(BOLD)Comandos disponiveis:$(NC)\n"
	@printf "  $(CYAN)make install$(NC)  - instala o uv, dependencias e hooks de pre-commit\n"
	@printf "  $(CYAN)make run-mlp$(NC)  - roda o treino+teste do MLP pros 5 datasets\n"
	@printf "  $(CYAN)make run-cnn$(NC)  - roda o treino+teste da CNN (Fashion MNIST)\n"
	@printf "  $(CYAN)make clean$(NC)    - apaga os arquivos gerados em saidas/\n"

install:
	@if ! command -v uv >/dev/null 2>&1; then \
		$(call run_with_spinner,Installing uv,curl -LsSf https://astral.sh/uv/install.sh | sh); \
	fi
	@$(call run_with_spinner,Installing dependencies,uv sync --upgrade)
	@$(call run_with_spinner,Setting up pre-commit hooks,uv run pre-commit install)
	@printf "$(BOLD)$(GREEN)  ✓ Done! Environment ready.$(NC)\n"
	@printf "\n$(DIM)  Next steps:$(NC)\n"
	@printf "$(DIM)    make run-mlp$(NC)\n"
	@printf "$(DIM)    make run-cnn$(NC)\n"

run-mlp:
	@printf "$(BOLD)$(CYAN)Rodando MLP (OR, AND, XOR, CARACTERES_REDUZIDO, CARACTERES_COMPLETO)...$(NC)\n\n"
	@uv run python src/mlp/main.py

run-cnn:
	@printf "$(BOLD)$(CYAN)Rodando CNN (Fashion MNIST)...$(NC)\n\n"
	@uv run python src/cnn/main.py

clean:
	@rm -rf saidas/*/
	@printf "$(GREEN)  ✓ saidas/ limpo$(NC)\n"
