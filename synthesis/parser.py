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
    'PLUS','MINUS','TIMES','DIVIDE','CEQ','EQ','POINT','GE','GT','LE','LT',
    'LPAREN','RPAREN','LBRACK','RBRACK','SEMICOL','COMMA'
    ] + list(reserved.values())

# Tokens

t_PLUS    = r'\+'
t_POINT   = r'->'
t_MINUS   = r'-'
t_TIMES   = r'\*'
t_DIVIDE  = r'/'
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
    ('left','GE','GT','LE','LT','CEQ'),
    ('left','PLUS','MINUS'),
    ('left','TIMES','DIVIDE'),
    )

# dictionary of names
names = { }

def p_program_one(t):
    'program : statement'
    #print t[1].strings()[0]
    ast = ASTNode()
    ast.addChild(t[1])
    t[0] = ast

def p_program_list(t):
    'program : program statement'
    #print t[2].strings()[0]
    t[1].addChild(t[2])
    t[0] = t[1]

def p_statement_vardecl(t):
    'statement : RANDOM ID ID TILDA expr SEMICOL'
    names[t[3]] = t[2]
    t[0] = VariableDeclNode(t[3],t[2],t[5])

################# Distributions ##################
def p_expr_bool(t):
    'expr : BOOLDIST LPAREN expr RPAREN'
    t[0] = BooleanDistribNode(None,t[3])

def p_expr_categorical(t):
    'expr : CATEGORICAL LPAREN LBRACK maplist RBRACK RPAREN'
    l = t[4]
    values = [x[0] for x in l]
    map = {}
    for x in l:
        map[x[0]] = x[1]
    t[0] = CategoricalDistribNode(None,values,map)

def p_expr_gauss(t):
    'expr : GAUSS LPAREN expr COMMA expr RPAREN'
    t[0] = GaussianDistribNode(None,t[3],t[5])

def p_expr_beta(t):
    'expr : BETA LPAREN expr COMMA expr RPAREN'
    t[0] = BetaDistribNode(None,t[3],t[5])

def p_expr_gamma(t):
    'expr : GAMMA LPAREN expr COMMA expr RPAREN'
    t[0] = GammaDistribNode(None,t[3],t[5])

def p_expr_uniform(t):
    'expr : UNIFORM LPAREN expr COMMA expr RPAREN'
    t[0] = UniformRealDistribNode(None,t[3],t[5])

##################################################

def p_expr_var(t):
    'expr : var'
    t[0] = t[1]

def p_var(t):
    'var : ID'
    if t[1] in names:
        t[0] = VariableUseNode(t[1],names[t[1]])
    else:
        t[0] = t[1]

def p_expr_const(t):
    'expr : NUMBER'
    t[0] = t[1]

def p_expr_boolbin(t):
    'expr : expr boolop expr'
    t[0] = BoolBinExpNode(t[2],t[1],t[3])

def p_boolop(t):
    '''boolop : GE
              | GT
              | LE
              | LT
              | CEQ'''
    t[0] = t[1]
    
def p_expr_if(t):
    'expr : IF expr THEN expr ELSE expr'
    t[0] = IfNode([t[2]],[t[4],t[6]])

def p_expr_case(t):
    'expr : CASE var IN LBRACK maplist RBRACK'
    l = t[5]
    print l
    cond = [ComparisonNode(t[2],"==",x[0]) for x in l[:-1]]
    body = [x[1] for x in l]
    t[0] = IfNode(cond,body)

def p_mappair(t):
    'mappair : ID POINT expr'
    t[0] = (t[1],t[3])

def p_maplist_unit(t):
    'maplist : mappair'
    t[0] = [t[1]]

def p_maplist_list(t):
    'maplist : maplist COMMA mappair'
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

####################################################

def p_error(t):
    print("Syntax error at '%s'" % t.value)

import ply.yacc as yacc
parser = yacc.yacc()

def parse(p):
    ast = parser.parse(p)
    print ast.strings()[0]

p1 = """
random Real tired ~ UniformReal(.5,.1);
random Real tired ~ Beta(.5,.1);
random Real tired ~ Gamma(.5,.1);
"""
parse(p1)
