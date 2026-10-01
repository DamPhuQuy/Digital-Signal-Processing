#!/usr/bin/env bash
# ==============================================================================
# Turnkey Environment Bootstrap & Baseline Verifier
# Part of RIPER-5 Engineering Harness (Learn Harness Engineering Standard)
# ==============================================================================
# Usage:
#   ./init.sh                     # Install deps, verify baseline, display start command
#   RUN_START_COMMAND=1 ./init.sh # Install deps, verify, and automatically run start command
# ==============================================================================

set -eo pipefail

# --- Project-specific configuration commands (adapt to your stack) ---
INSTALL_CMD="${INSTALL_CMD:-npm install}"
VERIFY_CMD="${VERIFY_CMD:-npm test}"
START_CMD="${START_CMD:-npm run dev}"

# --- Color formatting ---
BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RESET='\033[0m'

echo -e "${BOLD}${CYAN}=== [Harness Environment Bootstrap] ===${RESET}"
echo -e "${BOLD}Repository root:${RESET} $(pwd)"
echo -e "${BOLD}Timestamp:${RESET} $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo ""

# 1. Dependency installation
echo -e "${BOLD}${YELLOW}>>> Step 1: Installing dependencies...${RESET}"
echo -e "Command: ${CYAN}${INSTALL_CMD}${RESET}"
eval "$INSTALL_CMD"
echo -e "${GREEN}✓ Dependencies installed successfully.${RESET}\n"

# 2. Baseline verification
echo -e "${BOLD}${YELLOW}>>> Step 2: Running baseline verification...${RESET}"
echo -e "Command: ${CYAN}${VERIFY_CMD}${RESET}"
if eval "$VERIFY_CMD"; then
  echo -e "${GREEN}✓ Baseline verification PASSED. Environment is clean and ready.${RESET}\n"
else
  echo -e "${RED}✗ Baseline verification FAILED!${RESET}"
  echo -e "${RED}HALT: Agent must repair baseline failures before introducing any new changes.${RESET}"
  exit 1
fi

# 3. Start command handling
echo -e "${BOLD}${YELLOW}>>> Step 3: Application runtime command${RESET}"
echo -e "Start command: ${CYAN}${START_CMD}${RESET}"
if [ "${RUN_START_COMMAND:-0}" = "1" ]; then
  echo -e "${GREEN}Executing start command...${RESET}"
  exec eval "$START_CMD"
else
  echo -e "${BOLD}To start the service manually, execute:${RESET} ${CYAN}${START_CMD}${RESET}"
  echo -e "${GREEN}=== Bootstrap complete. Ready for task execution ===${RESET}"
fi
