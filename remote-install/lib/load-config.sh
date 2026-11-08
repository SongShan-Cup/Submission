# Source this file from scripts under remote-install/**.
_H3_INSTALL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
_H3_CONFIG_FILE="${H3_CONFIG_FILE:-$_H3_INSTALL_DIR/h3-paths.env}"
if [ ! -r "$_H3_CONFIG_FILE" ]; then
    printf '%s\n' "H3 config file is not readable: $_H3_CONFIG_FILE" >&2
    return 1
fi
. "$_H3_CONFIG_FILE"
for _H3_VAR in H3_PROJECT_ROOT H3_DEPS_ROOT H3_MODEL_ROOT H3_INPUT_IMAGE H3_DTK_ROOT H3_HYHAL_ROOT; do
    if [ -z "${!_H3_VAR:-}" ]; then
        printf '%s\n' "Missing $_H3_VAR in $_H3_CONFIG_FILE" >&2
        return 1
    fi
    case "${!_H3_VAR}" in
        /*) ;;
        *) printf '%s\n' "$_H3_VAR must be an absolute path: ${!_H3_VAR}" >&2; return 1 ;;
    esac
done
export H3_CONFIG_FILE="$_H3_CONFIG_FILE"
export H3_PROJECT_ROOT H3_DEPS_ROOT H3_MODEL_ROOT H3_INPUT_IMAGE H3_DTK_ROOT H3_HYHAL_ROOT
unset _H3_INSTALL_DIR _H3_CONFIG_FILE _H3_VAR
