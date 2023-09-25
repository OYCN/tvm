#ifndef TVM_TARGET_SOURCE_CODEGEN_RAWC_H_
#define TVM_TARGET_SOURCE_CODEGEN_RAWC_H_

#include <tvm/target/codegen.h>

#include "codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenRawC : public CodeGenC {
 public:
  using CodeGenC::PrintType;  // NOLINT(*)
  void PrintType(DataType t, std::ostream& os) final;
  void PrintCommon(const std::string& str);
  void VisitExpr_(const CallNode* op, std::ostream& os) override;  // NOLINT(*)
  Array<String> GetFunctionNames() { return function_names_; }

 private:
  struct FunctionInfo {
    /* \brief function name */
    std::string func_name;
    /* number of arguments required by the function */
    int64_t num_args;
    /* \brief name of resource_handle to pass */
    std::string resource_handle_name;
  };
  FunctionInfo GetFunctionInfo(const CallNode* op, bool has_resource_handle);
  std::string GetPackedName(const CallNode* op);
  Array<String> function_names_;
  std::unordered_map<std::string, std::string> declared_globals_;
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_RAWC_H_