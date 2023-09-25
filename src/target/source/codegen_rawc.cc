#include "codegen_rawc.h"

#include <tvm/relay/executor.h>
#include <tvm/relay/runtime.h>

namespace tvm {
namespace codegen {

void CodeGenRawC::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK_EQ(lanes, 1) << "does not support vector types";
    os << "void*";
    return;
  }
  if (t.is_void()) {
    os << "void";
    return;
  }
  if (t == DataType::Bool()) {
    os << "bool";
    return;
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        os << "half";
        break;
      case 32:
        os << "float";
        break;
      case 64:
        os << "double";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << 'u';
    }
    switch (t.bits()) {
      case 8:
        os << "int8_t";
        break;
      case 16:
        os << "int16_t";
        break;
      case 32:
        os << "int32_t";
        break;
      case 64:
        os << "int64_t";
        break;
      case 1:
        os << "int32_t";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes;
      return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to C type";
}

void CodeGenRawC::PrintCommon(const std::string& str) {
  this->PrintIndent();
  this->stream << "// " << str << "\n";
}

CodeGenRawC::FunctionInfo CodeGenRawC::GetFunctionInfo(const CallNode* op,
                                                       bool has_resource_handle) {
  const StringImmNode* s = op->args[0].as<StringImmNode>();
  ICHECK(s != nullptr) << "tvm_call_[c]packed_lowered expects first argument as function name";
  int64_t begin = op->args[3].as<IntImmNode>()->value;
  int64_t end = op->args[4].as<IntImmNode>()->value;
  int64_t num_args = end - begin;
  ICHECK_GE(num_args, 0);
  std::string func_name = s->value;

  if (has_resource_handle) {
    const StringImmNode* resource_handle_var = op->args[5].as<StringImmNode>();
    if (resource_handle_var != nullptr) {
      std::string resource_handle_name = resource_handle_var->value;
      return {func_name, num_args - 1, resource_handle_name};
    } else {
      // The final arg should be "(void*) NULL" to indicate the empty resource_handle.
      num_args--;

      const CallNode* reinterpret_call = op->args[5].as<CallNode>();
      ICHECK_NE(reinterpret_call, (void*)nullptr)
          << "At CallNode to " << s
          << "arg 5: Expect either StringImm naming the resource_handle var from interface API or "
          << "reinterpret(0); got: " << op->args[5];
      ICHECK_EQ(reinterpret_call->op, builtin::reinterpret())
          << "At CallNode to " << s
          << "arg 5: Expect either StringImm naming the resource_handle var from interface API or "
          << "reinterpret(0); got: " << op->args[5];
      ICHECK(is_zero(reinterpret_call->args[0])) << "At CallNode to " << s
                                                 << " arg 5: Expect either StringImm naming the "
                                                    "resource_handle var from interface API, or "
                                                 << "zero; got " << op->args[5];
    }
  }
  return {func_name, num_args, "NULL"};
}

std::string CodeGenRawC::GetPackedName(const CallNode* op) {
  const StringImmNode* s = op->args[0].as<StringImmNode>();
  ICHECK(s != nullptr) << "tvm_call_packed_lowered expects first argument as function name";
  std::string func_name = s->value;
  std::string packed_func_name = func_name + "_packed";
  std::string unique_name;
  auto it = declared_globals_.find(packed_func_name);
  if (it != declared_globals_.end()) {
    unique_name = it->second;
  } else {
    unique_name = name_supply_->FreshName(packed_func_name);
    declared_globals_[packed_func_name] = unique_name;
    decl_stream << "static void* " << unique_name << " = NULL;\n";
  }
  return unique_name;
}

void CodeGenRawC::VisitExpr_(const CallNode* op, std::ostream& os) {  // NOLINT(*)
  if (op->op.same_as(builtin::tvm_stack_alloca())) {
    std::string stack_name = name_supply_->FreshName("stack");
    const std::string& type = op->args[0].as<StringImmNode>()->value;
    const IntImmNode* num = op->args[1].as<IntImmNode>();
    ICHECK(num != nullptr);
    static_assert(alignof(TVMValue) % alignof(DLTensor) == 0, "invariant");
    size_t unit = sizeof(TVMValue);
    size_t size = 0;
    if (type == "shape") {
      size = (num->value * sizeof(tvm_index_t) + unit - 1) / unit;
    } else if (type == "arg_value") {
      size = (num->value * sizeof(TVMValue) + unit - 1) / unit;
    } else if (type == "arg_tcode") {
      size = (num->value * sizeof(int) + unit - 1) / unit;
    } else if (type == "array") {
      size = (num->value * sizeof(DLTensor) + unit - 1) / unit;
    } else {
      LOG(FATAL) << "Unknown stack alloca type " << type;
    }
    this->PrintIndent();
    this->stream << "TVMValue " << stack_name << "[" << size << "];\n";
    os << stack_name;
  } else if (op->op.same_as(builtin::tvm_call_packed_lowered())) {
    auto function_info = GetFunctionInfo(op, false /* has_resource_handle */);
    std::string func_name_packed = GetPackedName(op);
    this->PrintCommon("Call packed function");
    // this->PrintGetFuncFromBackend(function_info.func_name, func_name_packed);
    // this->PrintFuncCall(func_name_packed, function_info.num_args);
  } else if (op->op.same_as(builtin::tvm_call_cpacked_lowered())) {
    auto function_info = GetFunctionInfo(op, true /* has_resource_handle */);
    this->PrintCommon("Call packed function");
    // this->PrintFuncCallC(function_info.func_name, function_info.num_args,
    //                      function_info.resource_handle_name);
  } else if (op->op.same_as(builtin::tvm_throw_last_error())) {
    this->PrintIndent();
    this->stream << "return -1;\n";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

runtime::Module BulidRawC(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;

  std::unordered_set<std::string> devices;
  if (mod->GetAttr<Map<GlobalVar, String>>("device_contexts") != nullptr) {
    Map<GlobalVar, String> device_contexts =
        mod->GetAttr<Map<GlobalVar, String>>("device_contexts").value();
    for (auto const& context : device_contexts) {
      devices.insert(context.second.data());
    }
  }

  CodeGenRawC cg;
  cg.Init(output_ssa);
  cg.SetConstantsByteAlignment(target->GetAttr<Integer>("constants-byte-alignment").value_or(16));
  PrimFunc aot_executor_fn;

  std::vector<std::pair<tvm::GlobalVar, tvm::BaseFunc>> funcs;
  for (auto kv : mod->functions) {
    // Make sure that the executor function is the last one to be code generated so that all the
    // symbols are available to __tvm_main__
    auto fun_name = std::string(kv.first->name_hint);
    bool is_aot_executor_fn = kv.second->GetAttr<Bool>("runner_function", Bool(false)).value();

    if (is_aot_executor_fn) {
      aot_executor_fn = Downcast<PrimFunc>(kv.second);
      continue;
    }
    funcs.push_back(kv);
  }

  // Sort functions
  std::sort(funcs.begin(), funcs.end(),
            [](std::pair<tvm::GlobalVar, tvm::BaseFunc> kv_a,
               std::pair<tvm::GlobalVar, tvm::BaseFunc> kv_b) {
              std::string name_hint_a = kv_a.first->name_hint;
              std::string name_hint_b = kv_b.first->name_hint;
              return name_hint_a < name_hint_b;
            });

  // Add all functions except __tvm_main__
  for (auto& kv : funcs) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodegenCHost: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(f);
  }

  // Add __tvm_main__
  if (aot_executor_fn.defined()) {
    cg.AddFunction(aot_executor_fn);
  }

  if (target->GetAttr<Bool>("system-lib").value_or(Bool(false))) {
    ICHECK_EQ(target->GetAttr<String>("runtime").value_or(""), "c")
        << "c target only supports generating C runtime SystemLibs";
  }

  std::string code = cg.Finish();
  return CSourceModuleCreate(code, "c", cg.GetFunctionNames());
}

TVM_REGISTER_GLOBAL("target.build.rawc").set_body_typed(BulidRawC);
}  // namespace codegen
}  // namespace tvm
