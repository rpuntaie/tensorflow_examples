#!/usr/bin/env python3

# /tensorflow/python/autograph/g3doc/pyct_tutorial.md

#Requires `tf-nightly`

### Writing a custom code translator

import gast
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct import transpiler

class BasicCppCodegen(transformer.CodeGenerator):

  def visit_Name(self, node):
    self.emit(node.id)

  def visit_arguments(self, node):
    self.visit(node.args[0])
    for arg in node.args[1:]:
      self.emit(', ')
      self.visit(arg)

  def visit_FunctionDef(self, node):
    self.emit('void {}'.format(node.name))
    self.emit('(')
    self.visit(node.args)
    self.emit(') {\n')
    self.visit_block(node.body)
    self.emit('\n}')

  def visit_Call(self, node):
    self.emit(node.func.id)
    self.emit('(')
    self.visit(node.args[0])
    for arg in node.args[1:]:
      self.emit(', ')
      self.visit(arg)
    self.emit(');')

class PyToBasicCpp(transpiler.GenericTranspiler):

  def transform_ast(self, node, ctx):
    codegen = BasicCppCodegen(ctx)
    codegen.visit(node)
    return codegen.code_buffer

# f needs to be in a python file
def f1(x, y):
  print(x, y)

def convert1():
  code, _ = PyToBasicCpp().transform(f1, None)
  print(code)


#The `static_analysis` module contains various helper passes for dataflow analyis.
#All these passes annotate the AST. These annotations can be extracted using
#anno.getanno.  Most of them rely on the `qual_names` annotations, which just
#simplify the way more complex identifiers like `a.b.c` are accessed.
# The most useful is the activity analysis which just inventories symbols read, modified, etc.:

def get_node_and_ctx(f):
  node, source = parser.parse_entity(f, ())
  f_info = transformer.EntityInfo(
    name='f',
    source_code=source,
    source_file=None,
    future_features=(),
    namespace=None)
  ctx = transformer.Context(f_info, None, None)
  return node, ctx

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct.static_analysis import annos
from tensorflow.python.autograph.pyct.static_analysis import activity


def f2(a):
  b = a + 1
  return b

def convert2():
    node, ctx = get_node_and_ctx(f2)

    node = qual_names.resolve(node)
    node = activity.resolve(node, ctx)

    fn_scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)  # Note: tag will be changed soon.

    print('read:', fn_scope.read)
    print('modified:', fn_scope.modified)

#Another useful utility is the control flow graph builder.
#Of course, a CFG that fully accounts for all effects is impractical to build in
#a late-bound language like Python without creating an almost fully-connected
#graph. However, one can be reasonably built if we ignore the potential for
#functions to raise arbitrary exceptions.

from tensorflow.python.autograph.pyct import cfg

def f3(a):
  if a > 0:
    return a
  b = -a


def convert3():
    node, ctx = get_node_and_ctx(f3)
    node = qual_names.resolve(node)
    cfgs = cfg.build(node)
    cfgs[node]

#Other useful analyses include liveness analysis. Note that these make
#simplifying assumptions, because in general the CFG of a Python program is a
#graph that's almost complete. The only robust assumption is that execution
#can't jump backwards.

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct.static_analysis import annos
from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions
from tensorflow.python.autograph.pyct.static_analysis import reaching_fndefs
from tensorflow.python.autograph.pyct.static_analysis import liveness

def f4(a):
  b = a + 1
  return b

def convert4():
    node, ctx = get_node_and_ctx(f4)

    node = qual_names.resolve(node)
    cfgs = cfg.build(node)
    node = activity.resolve(node, ctx)
    node = reaching_definitions.resolve(node, ctx, cfgs)
    node = reaching_fndefs.resolve(node, ctx, cfgs)
    node = liveness.resolve(node, ctx, cfgs)

    print('live into `b = a + 1`:', anno.getanno(node.body[0], anno.Static.LIVE_VARS_IN))
    print('live into `return b`:', anno.getanno(node.body[1], anno.Static.LIVE_VARS_IN))

### Custom Python-to-Python transpiler
#`transpiler.Py2Py` is a generic class for a Python 
#[source-to-source compiler](https://en.wikipedia.org/wiki/Source-to-source_compiler).
#It operates on Python ASTs. Subclasses override its transform_ast method.
#Unlike the `transformer` module, which have an AST as input/output, the
#`transpiler` APIs accept and return actual Python objects, handling the tasks
#associated with parsing, unparsing and loading of code.


from tensorflow.python.autograph.pyct import transpiler

class NoopTranspiler(transpiler.PyToPy):

  def get_caching_key(self, ctx):
    # You may return different caching keys if the transformation may generate
    # code versions.
    return 0

  def get_extra_locals(self):
    # No locals needed for now; see below.
    return {}

  def transform_ast(self, ast, transformer_context):
    return ast

tr = NoopTranspiler()

#The main entry point is ``transform``.

def f5(x, y):
  return x + y

def convert5():
    new_f, module, source_map = tr.transform(f5, None)
    new_f(1, 1)

### Adding new variables to the transformed code
#The transformed function has the same global and local variables as the
#original function. You can of course generate local imports to add any new
#references into the generated code, but an easier method is to use the
#`get_extra_locals` method:

from tensorflow.python.autograph.pyct import parser

class HelloTranspiler(transpiler.PyToPy):

  def get_caching_key(self, ctx):
    return 0

  def get_extra_locals(self):
    return {'name': 'you'}

  def transform_ast(self, ast, transformer_context):
    print_code = parser.parse('print("Hello", name)')
    ast.body = [print_code] + ast.body
    return ast


def f6(x, y):
  pass

def convert6():
    new_f, _, _ = HelloTranspiler().transform(f6, None)
    _ = new_f(1, 1)
    import inspect
    print(inspect.getsource(new_f))


def main():
  convert1()
  convert2()
  convert3()
  convert4()
  convert5()
  convert6()

if __name__ == "__main__":
    main()

