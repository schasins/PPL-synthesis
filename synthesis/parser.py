class ParserError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

reserved = {
    'random'   : 'RANDOM',
    # 'true'  : 'TRUE',
    # 'false' : 'FALSE',
    'BooleanDistrib' : 'BOOLDIST',
    'Categorical' : 'CATEGORICAL',
    'Gaussian' : 'GAUSS',
    'Beta' : 'BETA',
    'Gamma' : 'GAMMA',
    'UniformReal' : 'UNIFORM',
    'if'   : 'IF',
    'then' : 'THEN',
    'else' : 'ELSE',
    'case' : 'CASE',
    'in'   : 'IN',
    'distinct' : 'DISTINCT'
}

tokens = [
    'TILDA', 
    'ID','NUMBER',
    'PLUS','MINUS','TIMES','DIVIDE','AND','OR',
    'CEQ','EQ','POINT','GE','GT','LE','LT',
    'LPAREN','RPAREN','LBRACK','RBRACK','LSQBK','RSQBK',
    'SEMICOL','COMMA'
    ] + list(reserved.values())

# Tokens

t_PLUS    = r'\+'
t_POINT   = r'->'
t_MINUS   = r'-'
t_TIMES   = r'\*'
t_DIVIDE  = r'/'
t_AND     = r'\&'
t_OR      = r'\|'
t_GE      = r'<='
t_GT      = r'<'
t_LE      = r'>='
t_LT      = r'>'
t_CEQ      = r'=='
t_EQ      = r'='
t_TILDA   = r'~'
t_LPAREN  = r'\('
t_RPAREN  = r'\)'
t_LBRACK  = r'\{'
t_RBRACK  = r'\}'
t_LSQBK  = r'\['
t_RSQBK  = r'\]'
t_SEMICOL = r'\;'
t_COMMA   = r'\,'

def t_ID(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value,'ID')    # Check for reserved words
    return t

def t_NUMBER(t):
    r'\d*[.]\d+ | \d+'
    try:
        t.value = float(t.value)
    except ValueError:
        print("Integer value too large %d", t.value)
        t.value = 0
    return t

# Ignored characters
t_ignore = " \t"

def t_ignore_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")

def t_ignore_comment(t):
    r'(/\*(.|\n)*?\*/)|(//.*)'
    pass
    
def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)
    
# Build the lexer
import ply.lex as lex
lexer = lex.lex()
from ast import *

# Parsing rules
precedence = (
    ('left','AND','OR'),
    ('left','GE','GT','LE','LT','CEQ'),
    ('left','PLUS','MINUS'),
    ('left','TIMES','DIVIDE'),
    )

# dictionary of names
names = { }

def p_program_one(t):
    'program : statement'
    ast = ASTNode()
    ast.addChild(t[1])
    t[0] = ast

def p_program_list(t):
    'program : program statement'
    t[1].addChild(t[2])
    t[0] = t[1]

def p_statement_vardecl(t):
    'statement : RANDOM ID ID TILDA expr SEMICOL'
    names[t[3]] = t[2]
    t[0] = VariableDeclNode(t[3],t[2],t[5])

################# Distributions ##################
def p_expr_bool(t):
    'expr : BOOLDIST LPAREN NUMBER RPAREN'
    t[0] = BooleanDistribNode(None,t[3])

def p_expr_categorical(t):
    'expr : CATEGORICAL LPAREN LBRACK pairs RBRACK RPAREN'
    l = t[4]
    values = [x[0] for x in l]
    map = {}
    for x in l:
        map[x[0]] = x[1]
    t[0] = CategoricalDistribNode(None,values,map)

def p_expr_gauss(t):
    'expr : GAUSS LPAREN NUMBER COMMA NUMBER RPAREN'
    t[0] = GaussianDistribNode(None,t[3],t[5])

def p_expr_beta(t):
    'expr : BETA LPAREN NUMBER COMMA NUMBER RPAREN'
    t[0] = BetaDistribNode(None,t[3],t[5])

def p_expr_gamma(t):
    'expr : GAMMA LPAREN NUMBER COMMA NUMBER RPAREN'
    t[0] = GammaDistribNode(None,t[3],t[5])

def p_expr_uniform(t):
    'expr : UNIFORM LPAREN NUMBER COMMA NUMBER RPAREN'
    t[0] = UniformRealDistribNode(None,t[3],t[5])

def p_pair(t):
    'pair : ID POINT NUMBER'
    t[0] = (t[1],t[3])

def p_pairs_unit(t):
    'pairs : pair'
    t[0] = [t[1]]

def p_pairs_list(t):
    'pairs : pairs COMMA pair'
    t[1].append(t[3])
    t[0] = t[1]

##################################################

def p_expr_var(t):
    'expr : var'
    t[0] = t[1]

def p_var(t):
    'var : ID'
    if t[1] in names:
        t[0] = VariableUseNode(t[1],names[t[1]])
    elif t[1] == "true":
        t[0] = BooleanDistribNode(None,1)
    elif t[1] == "false":
        t[0] = BooleanDistribNode(None,0)
    else:
        t[0] = StringValue(t[1])

def p_expr_const(t):
    'expr : NUMBER'
    t[0] = NumericValue(t[1])

def p_expr_bin(t):
    '''expr : boolbin
            | arithbin
            | compare'''
    t[0] = t[1]

def p_expr_binparen(t):
    '''expr : LPAREN boolbin RPAREN
            | LPAREN arithbin RPAREN
            | LPAREN compare RPAREN'''
    t[0] = t[2]
    
def p_compare(t):
    '''compare : expr GE expr
               | expr GT expr
               | expr LE expr
               | expr LT expr
               | expr CEQ expr'''
    #print "COMPARE", t[1], t[2], t[3]
    t[0] = CreateComparisonNode(t[1],t[2],t[3])
    
def p_boolbin(t):
    '''boolbin : expr AND expr
               | expr OR expr'''
    #print "BOOLBIN", t[1], t[2], t[3]
    t[0] = BoolBinExpNode(t[2],t[1],t[3])
    
def p_arithbin(t):
    '''arithbin : expr PLUS expr
           | expr MINUS expr
           | expr TIMES expr
           | expr DIVIDE expr'''
    #print "ARITHBIN", t[1], t[2], t[3]
    t[0] = BinExpNode(t[2],t[1],t[3])
    
def p_expr_if(t):
    'expr : IF expr THEN expr ELSE expr'
    t[0] = IfNode([t[2]],[t[4],t[6]])

def p_expr_case(t):
    'expr : CASE expr IN LBRACK exprpairs RBRACK'
    l = t[5]
    #print "CASE", t[2], l
    cond = [CreateComparisonNode(t[2],"==",x[0]) for x in l[:-1]]
    body = [x[1] for x in l]
    t[0] = IfNode(cond,body)

def p_exprpair(t):
    'exprpair : expr POINT expr'
    t[0] = (t[1],t[3])

def p_exprpairs_unit(t):
    'exprpairs : exprpair'
    t[0] = [t[1]]

def p_exprpairs_list(t):
    'exprpairs : exprpairs COMMA exprpair'
    t[1].append(t[3])
    t[0] = t[1]

def p_expr_tuble(t):
    'expr : LSQBK varlist RSQBK'
    t[0] = t[2]

def p_varlist_unit(t):
    'varlist : var'
    t[0] = [t[1]]

def p_varlist_list(t):
    'varlist : varlist COMMA var'
    t[1].append(t[3])
    t[0] = t[1]

################# Type declaration #################

def p_statement_type(t):
    'statement : DISTINCT ID idlist SEMICOL'
    t[0] = TypeDeclNode(t[2],t[3])

def p_idlist_unit(t):
    'idlist : ID'
    t[0] = [t[1]]

def p_idlist_list(t):
    'idlist : idlist COMMA ID'
    t[1].append(t[3])
    t[0] = t[1]

################# Helper functions ##################
def CreateComparisonNode(lhs,op,rhs):
    if isinstance(lhs,ASTNode) and isinstance(rhs,ASTNode):
        return ComparisonNode(lhs,op,rhs)
    elif isinstance(lhs,list) and isinstance(rhs,list) and len(lhs) == len(rhs):
        l = []
        for i in range(len(lhs)):
            l.append(ComparisonNode(lhs[i],op,rhs[i]))
        ast = BoolBinExpNode('&',l[0],l[1])
        for i in range(2,len(l)):
            ast = BoolBinExpNode('&',ast,l[i])
        return ast
    else:
        raise ParserError("Fail to create ComparisonNode for" + str(lhs) + str(op) + str(rhs))

def p_error(t):
    print("Syntax error at '%s'" % t.value)

import ply.yacc as yacc
parser = yacc.yacc()

def parse_from_file(name):
    f = open(name,'r')
    ast = parser.parse(f.read())
    f.close()
    #print ast.strings()[0]
    return ast

if __name__ == "__main__":
    p = """
random Boolean HealthConscious ~ BooleanDistrib(0.5);
random Boolean LittleFreeTime ~ BooleanDistrib(0.5);
random Boolean Exercise
  ~ case [HealthConscious, LittleFreeTime] in
               {[true, true] -> BooleanDistrib(0.5),
                [true, false] -> BooleanDistrib(0.9),
                [false, true] -> BooleanDistrib(0.1),
                [false, false] -> BooleanDistrib(0.5)}
    ;
random Boolean GoodDiet
  ~ case HealthConscious in {true -> BooleanDistrib(0.7),
                false -> BooleanDistrib(0.3)};
random Boolean NormalWeight 
  ~ case [GoodDiet, Exercise] in 
               {[true, true] -> BooleanDistrib(0.8),
                [true, false] -> BooleanDistrib(0.5),
                [false, true] -> BooleanDistrib(0.5),
                [false, false] -> BooleanDistrib(0.2)}
    ;
random Boolean HighCholesterol
  ~ case GoodDiet in 
                {true -> BooleanDistrib(0.3),
                 false -> BooleanDistrib(0.7)}
    ;
random Boolean TestedHighCholesterol
  ~ case HighCholesterol in {true -> BooleanDistrib(0.9),
               false -> BooleanDistrib(0.1)}
    ;
"""
    ast = parser.parse(p)
    print ast.strings()[0]
