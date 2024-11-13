---
layout:     post
title:      "Kalami"
subtitle:   "A Toy Non-Deterministic Programming Language"
date:       2024-06-07 12:00:00
author:     "Morten Dahl"
header-img: "img/post-bg-04.jpg"
---

<em><strong>TL;DR:</strong> TODO</em>

This is a revisit of a blog post I wrote more than ten years ago, back in 2013. Specifically, during my years in academia, I was a teaching assistant in a course on [computability](http://en.wikipedia.org/wiki/Computability), and as such involved in explaining the concept of [non-deterministic “guessing”](https://en.wikipedia.org/wiki/Nondeterministic_Turing_machine). I always found this topic fascinating, so since I also had to come up with a few project suggestions for the students, I thought some of them might be interested in making a toy programming language with a guessing construct. To give them a place to start, I quickly put together a prototype interpreter in [OCaml](http://caml.inria.fr/) that is the topic of this blog post.

# Introduction

Say you want to find two factors in a given integer `n`. There are many algorithms for doing this, but in a non-determinism computational model we may simply ask the computer to “guess” them. Expressed in the toy language we can write:

```ocaml
let n = 681 in

guess p from 2 in
guess q from 2 in

if n == p*q then accept else reject
```

Here `p` and `q` are the factors to be guessed, and we assume they are larger than 2 to avoid the trivial solutions where one of them is equal to 1.

TODO: sat solver. Conjecture where there is no obvious upper bound.

TODO pure language so we can reevaluate

This could of course be expressed similarly in e.g. [Prolog](http://en.wikipedia.org/wiki/Prolog), but the aim was also for the project to have a touch of finding suitable evaluation strategies and using static analysis, as well as to illustrate the power of working with languages in OCaml.

To illustrate some of the issues consider program

```ocaml
guess x in
guess y in
if x == y+1 then accept else reject
```

which have solution x=1 and y=0. Yet if the evaluation strategy simply starts with x=0 and tries all values for y then it will never find it; in other words, the evaluation strategy must eventually enumerate all tuples of values.

For some programs we may also benefit from first employing static analysis. Consider for instance program

```ocaml
guess x in
if x == x+1 then accept else reject
```

that is easily seen to not have any solutions — yet the interpreter might not realise this without some kind of static analyse. Likewise, for programs such as

```ocaml
guess n in
if n == 1*2*3 then accept else reject
```

static analysis may speed up execution time by letting the interpreter discover that it only needs to consider a subset of values — in this case only n=6 — instead of it simply brute forcing for all possibly values.

# Parsing

Next up is parsing. I remember how we used [SableCC](http://sablecc.org/) during my undergraduate years, and I also remember how we simply couldn’t make it pretty no matter what measure we used! Perhaps it can be made better in the Java environment than we ever managed to, but after I saw how it is done in the OCaml environment I never looked back: all of a sudden it became fun to write parsers and interpreters!

So, parsing is basically about how to turn a input text string into an in-memory representation in the form of an [abstract syntax tree](http://en.wikipedia.org/wiki/Abstract_syntax_tree). I don’t want to go into a lot of background but a good source for slightly more details the [OCaml Manual](https://ocaml.org/manual/).

The files needed can be found in the ZIP file kalami-parsing.zip; out of these the most important are structure.ml, lexer.mll, and parser.mly.

Starting with structure.ml we first have a piece of OCaml code defining what we would like the abstract syntax tree of a Kalami program to look like:

```ocaml
type identifier = string
type number = int
type bound = int

type expression =
    ExprNumber of number
    | ExprVariable of identifier
    | ExprProduct of expression * expression
    | ExprDivision of expression * expression
    | ExprAddition of expression * expression
    | ExprSubtraction of expression * expression
    | ExprPower of expression * expression

type condition =
    CndTrue
    | CndFalse
    | CndEqual of expression * expression
    | CndLessthan of expression * expression
    | CndNegation of condition
    | CndOr of condition * condition
    | CndAnd of condition * condition

type statement =
    StmtAccept
    | StmtReject
    | StmtLet of identifier * expression * statement
    | StmtIf of condition * statement * statement
    | StmtGuess of identifier * statement
    | StmtGuessLowerBound of identifier * expression * statement
    | StmtGuessLowerUpperBound of identifier * expression * expression * statement
```

Essentially, the above code defines expressions, conditions, and statements as labelled tuples, such that when we for instance write “4*5” in Kalami then this can be represented in memory as expression `ExprProduct(ExprNumber(4), ExprNumber(5))`. We shall later see that in the end a Kalami program is simply a statement.

By the way, note the elegance with which we can express an abstract syntax tree in Caml: were we to express the same in Java it would look a lot less intuitive, at least in my experience. This is one of the great strengths of Caml, of which there are plenty more when we start evaluating and analysing the abstract syntax tree in later posts.

Next up is the lexer responsible for the first layer of abstraction of the input, namely turning the raw string into a stream of tokens. Looking in lexer.mll we see that several regular expressions are used to abstract strings, from the very basic labelling of characters and keywords:

```
    | '('           { LPAREN }
    | ')'           { RPAREN }
    | '+'           { PLUS }
    | '-'           { MINUS }

    | "let"         { LET }
    | "in"          { IN }
    | "if"          { IF }
    | "then"        { THEN }
    | "else"        { ELSE }
```

to the slightly more advanced regular expressions, that also includes a native OCaml instruction for turning a string into a number:

```
    | (['0'-'9']+ as i)     { INT(int_of_string i) }
```

Besides this abstraction of strings as tokens not much happens in the lexer, perhaps apart from the little extra code needed for allowing nested comments, and the insistence that STRs, and in turn identifiers, start with a digit (an easy way to distinguish them from INTs).

However, in parser.mly we may use these tokens to put together a [grammar](http://en.wikipedia.org/wiki/Formal_grammar). In particular we see from rule main that an input (now a string of tokens) is a Kalami program if it can be parsed according to rule “stmt” and its sub-rules as given by:

```
id:
    STR         { $1 }
;

expr:
    INT                     { ExprNumber($1) }
    | id                    { ExprVariable($1) }
    | expr MULT expr        { ExprProduct($1, $3) }
    | expr DIV expr         { ExprDivision($1, $3) }
    | expr PLUS expr        { ExprAddition($1, $3) }
    | expr MINUS expr       { ExprSubtraction($1...) }
    | expr EXP expr         { ExprPower($1, $3) }
    | LPAREN expr RPAREN    { $2 }
;

cnd:
    expr EQ expr            { CndEqual($1, $3) }
    | expr LT expr          { CndLessthan($1, $3) }
    | NOT cnd               { CndNegation($2) }
    | cnd OR cnd            { CndOr($1, $3) }
    | cnd AND cnd           { CndAnd($1, $3) }
;

stmt:
    ACCEPT                          { StmtAccept }
    | REJECT                        { StmtReject }
    | LET id ASSIGN expr IN stmt    { StmtLet($2, $4, $6) }
    | IF cnd THEN stmt ELSE stmt    { StmtIf($2, $4, $6) }
    | GUESS id IN stmt              { StmtGuess($2, $4) }
    | GUESS id FROM expr IN stmt    { StmtGuessLowerBo... }
    | GUESS id FROM expr TO         { StmtGuessLowerUp... }
;
```

Besides these rules the file also contains instructions defining the precedence of for example the mathematical operations as well as their associativity, as needed to help the parser solve ambiguity.

Having looked at the primary files for the parser we turn to the plumbing needed to put them together. File printing.ml simply takes a statement and recursively prints it on the screen, and file kalamiparsing.ml contains Caml code that reads the input, invokes the lexer and the parser, and prints the statement in case no error occurred:

```ocaml
open Printing

let _ =
    try
        let lexbuf = Lexing.from_channel stdin in
        let stmt = Parser.main Lexer.token lexbuf in

        print_string "\n*** Statement ***\n\n";
        print_stmt stmt;
        print_string "\n\n"

    with
        Lexer.Eof ->
            print_string "End of file\n";
            exit 0
        | Parsing.Parse_error ->
            print_string "Parsing error\n";
            exit 1
```

The only thing missing now is how to compile everything. This is not completely straight-forward since the lexer and the parser are referencing each other, and as a result we must execute the compilation commands in the specific order as seen in file compile.sh. Doing so will produce the kalamiparsing binary which we can tell to either load a program from a file through

./kalamiparsing < "inputfile"
or run “interactively” by simply executing

./kalamiparsing
and typing a program directly into it (ending with an EOF signal by pressing control + d).

In the next post we’ll look at how we may evaluate an in-memory Kalami program in order to find a satisfying guess; as mentioned previously this requires something slightly more involved than a trivial brute-force.

# Working with the Syntax Tree

While the grammar from the previous post on parsing puts some restrictions on the form of a valid Kalami program, it doesn’t reject all invalid programs. In particular, it doesn’t ensure that all variables are bound (declared) before use. Defining a simple type system to check for this is straight forward, but will also serve to illustrate how easy it is to work with an abstract syntax tree in Caml.

To quickly illustrate the problem, consider programs such as:

```
if x == 5 then accept else reject
```

which are meaningless since x is unbound. Now, since I don’t know any good way to present the rules of the type system here on the blog in the usual visual style of inference rules, I’m just going to give the code implementing them. Fortunately though, Caml allows us to express such rules so concisely that not much is lost this way anyway — a legacy of [ML](http://en.wikipedia.org/wiki/ML_(programming_language)) (its distant father) being a language for formulating logical rules for theorem provers!

Starting with expressions, we assume that a list of variables have already been defined (by outer statements) and by recursion on the structure of the expression check that all variables it mentions are in the list. The code looks as follows:

```ocaml
let rec wellformed_expr definedvars expr =
    match expr with

        ExprNumber(n) ->
            true

        | ExprVariable(id) ->
            List.mem id definedvars

        | ExprProduct(expr1, expr2) ->
            (wellformed_expr definedvars expr1) &&
            (wellformed_expr definedvars expr2)

        | ExprDivision(expr1, expr2) ->
            (wellformed_expr definedvars expr1) &&
            (wellformed_expr definedvars expr2)

        | ExprAddition(expr1, expr2) ->
            (wellformed_expr definedvars expr1) &&
            (wellformed_expr definedvars expr2)

        | ExprSubtraction(expr1, expr2) ->
            (wellformed_expr definedvars expr1) &&
            (wellformed_expr definedvars expr2)

        | ExprPower(expr1, expr2) ->
            (wellformed_expr definedvars expr1) &&
            (wellformed_expr definedvars expr2)
```

with the most interesting case being for `ExprVariable` where a call to standard library function `List.mem` checks the id is a member of the list. Checking a condition is similar, again assuming a list of variables that have already been bound; notice that it calls the above function for cases `CndEqual` and `CndLessthan`:

```ocaml
let rec wellformed_cnd definedvars cnd =
    match cnd with

        CndTrue ->
            true

        | CndFalse ->
            true

        | CndEqual(expr1, expr2) ->
            (wellformed_expr definedvars expr1) &&
            (wellformed_expr definedvars expr2)

        | CndLessthan(expr1, expr2) ->
            (wellformed_expr definedvars expr1) &&
            (wellformed_expr definedvars expr2)

        | CndNegation(cnd) ->
            wellformed_cnd definedvars cnd

        | CndOr(cnd1, cnd2) ->
            (wellformed_cnd definedvars cnd1) &&
            (wellformed_cnd definedvars cnd2)

        | CndAnd(cnd1, cnd2) ->
            (wellformed_cnd definedvars cnd1) &&
            (wellformed_cnd definedvars cnd2)
```

Now, for statements it finally gets slightly more interesting. The function still takes a list definedvars as input since it calls itself recursively, and hence some variables may be bound by outer statements. But it now also extends this list when a let– or guess-statement is encountered; in these cases it also ensures that no variable is re-bound:

```ocaml
let rec wellformed_stmt definedvars stmt =
    match stmt with

        StmtAccept ->
            true

        | StmtReject ->
            true

        | StmtLet(id, expr, stmt) ->
            (not (List.mem id definedvars)) &&
            (wellformed_expr definedvars expr) && 
            (wellformed_stmt (id::definedvars) stmt)

        | StmtIf(cnd, stmt_true, stmt_false) ->
            (wellformed_cnd definedvars cnd) &&
            (wellformed_stmt definedvars stmt_true) &&
            (wellformed_stmt definedvars stmt_false)

        | StmtGuess(id, stmt) ->
            (not (List.mem id definedvars)) &&
            (wellformed_stmt (id::definedvars) stmt)

        | StmtGuessLowerBound(id, _, stmt) ->
            (not (List.mem id definedvars)) &&
            (wellformed_stmt (id::definedvars) stmt)

        | StmtGuessLowerUpperBound(id, _, _, stmt) ->
            (not (List.mem id definedvars)) &&
            (wellformed_stmt (id::definedvars) stmt)
```

The above functions are all that is needed to perform our validation check: simply invoke `wellformed_stmt` on the main program statement with an empty list, and check that it returns true. We may wrap this in order to hide some of the implementation details:

```ocaml
let wellformed stmt =
    wellformed_stmt [] stmt
```

# Evaluation

It’s been a while now since the last post in my little series on Kalami, but I’ve finally polished the code for evaluating programs as I wanted to. Overall, it’s not that different from the wellformed code of the previous post; in fact, the most challenging aspect is how to enumerate all guesses.

As earlier, we’re dealing with statements, conditions, and expressions. The first one is the tricky one, so to start off simple I’ll first look at the other two. The full source code is available in the evalwoa branch on GitHub.

Assuming bindings in environment env for all identifiers in an expression expr, it may be evaluated straight-forwardly by the following recursive function `eval_expr`:

```ocaml
let rec eval_expr env expr =
    match expr with

        ExprNumber(n) ->
            n

        | ExprVariable(id) ->
            List.assoc id env

        | ExprProduct(expr1, expr2) ->
            let val1 = eval_expr env expr1 in
            let val2 = eval_expr env expr2 in
            val1 * val2

        | ExprDivision(expr1, expr2) ->
            let val1 = eval_expr env expr1 in
            let val2 = eval_expr env expr2 in
            val1 / val2

        | ExprAddition(expr1, expr2) ->
            let val1 = eval_expr env expr1 in
            let val2 = eval_expr env expr2 in
            val1 + val2

        | ExprSubtraction(expr1, expr2) ->
            let val1 = eval_expr env expr1 in
            let val2 = eval_expr env expr2 in
            val1 - val2

        | ExprPower(expr1, expr2) ->
            let val1 = eval_expr env expr1 in
            let val2 = eval_expr env expr2 in
            power val1 val2
```

Note first that since env is just a list of pairs binding identifiers to integers

```ocaml
let env = [ (id1, n1); (id2, n2); ... (idk, nk) ];;
```

we may extract the latter using built-in function `List.assoc` as in case `ExprVariable`. Next, note that there’s no built-in function for integer exponentiation, hence the `power` function used in case `ExprPower` is implemented in evaluation.ml as a tail recursive repeated squaring.

Evaluating condition `cnd` is similar, with the small difference that for one reason or another (purist perhaps) I used the prefix notation for function invocation here:

```ocaml
let rec eval_cnd env cnd =
    match cnd with

        CndTrue ->
            true

        | CndFalse ->
            false

        | CndEqual(expr1, expr2) ->
            let val1 = eval_expr env expr1 in
            let val2 = eval_expr env expr2 in
            (=) val1 val2

        | CndLessthan(expr1, expr2) ->
            let val1 = eval_expr env expr1 in
            let val2 = eval_expr env expr2 in
            (<) val1 val2

        | CndNegation(cnd) ->
            let val1 = eval_cnd env cnd in
            (not) val1

        | CndOr(cnd1, cnd2) ->
            let val1 = eval_cnd env cnd1 in
            let val2 = eval_cnd env cnd2 in
            (||) val1 val2

        | CndAnd(cnd1, cnd2) ->
            let val1 = eval_cnd env cnd1 in
            let val2 = eval_cnd env cnd2 in
            (&&) val1 val2
```

For statement stmt it finally gets a bit more interesting. Cases `StmtLet` and `StmtIf` are straight-forward, as are `StmtAccept` and `StmtReject` which respectively return the current environment and `None`. However, for a guessing statement it’s a bit more involved. If the guess has an upper bound then we may just try all possibilities as in case `StmtGuessLowerUpperBound` below. But this will of course not work when no upper bound is specified since an inner guess will prevent outer guesses from ever increasing. As an example, consider program:

```
guess x in
guess y in
if x == y+1 then accept
otherwise reject
```

which has solution [ ("x", 1); ("y", 0) ], yet if we first let x=0 and then use the strategy of enumerating all values for y then we’ll never make it back to incrementing x.

Instead, for a program with n (unbounded) guesses, i.e. guesses without an upper bound, the simple strategy used here is to enumerate all n-arity integer tuples in an outer loop, and for each evaluate the main statement with a second environment guesses linking the tuples with the identifiers of the (unbounded) guesses. Statements may then be evaluated as follows, using an initial empty environment:

```ocaml
let rec eval_stmt env guesses stmt =
    match stmt with

        StmtAccept ->
            Some(env)

        | StmtReject ->
            None

        | StmtLet(id, expr, stmt) ->
            let stmt_value = eval_expr env expr in
            let env' = List.cons (id, stmt_value) env in
            eval_stmt env' guesses stmt

        | StmtIf(cnd, stmt_true, stmt_false) ->
            let cnd_value = eval_cnd env cnd in
            if cnd_value then
                eval_stmt env guesses stmt_true
            else
                eval_stmt env guesses stmt_false

        | StmtGuess(id, stmt) ->
            let guess = List.assoc id guesses in
            let env' = List.cons (id, guess) env in
            eval_stmt env' guesses stmt

        | StmtGuessLowerBound(id, lower_bound_expr, stmt) ->
            let lower_bound = eval_expr env lower_bound_expr in
            let guess = List.assoc id guesses in
            if guess < lower_bound then
                  None
             else
                 let env' = List.cons (id, guess) env in
                 eval_stmt env' guesses stmt

        | StmtGuessLowerUpperBound(id, lb_expr, ub_expr, stmt) ->
            let lower_bound = eval_expr env lb_expr in
            let upper_bound = eval_expr env ub_expr in
            let rec helper guess =
                let env' = List.cons (id, guess) env in
                let result = eval_stmt env' guesses stmt in
                match result with
                    None ->
                        if guess < upper_bound then
                            helper (guess + 1)
                        else
                            None
                    | Some(_) ->
                        result
            in
            helper lower_bound
```

where we see that in cases `StmtGuess` and `StmtGuessLowerBound` the guess performed in the outer enumeration is simply added to the environment.

The only thing missing is hence the outer enumeration. First we have a simple function collecting identifiers:

```ocaml
let rec get_unbounded_guesses stmt =
    match stmt with

        StmtAccept ->
            []

        | StmtReject ->
            []

        | StmtLet(_, _, stmt) ->
            get_unbounded_guesses stmt

        | StmtIf(_, stmt_true, stmt_false) ->
            (get_unbounded_guesses stmt_true) 
            @ (get_unbounded_guesses stmt_false)

        | StmtGuess(id, stmt) ->
            id :: (get_unbounded_guesses stmt)

        | StmtGuessLowerBound(id, _, stmt) ->
            id :: (get_unbounded_guesses stmt)

        | StmtGuessLowerUpperBound(_, _, _, stmt) ->
            get_unbounded_guesses stmt
```

Next is a projection function project arity number performing a one-to-one mapping of an integer into a integer tuple with the given arity (see also projection.ml). The code for this gets a bit messy so I won’t give it here, yet the main principle is to partition the digits in number such that for instance:

```ocaml
# project 3 123456;;
- : (int * int) list = [(3, 14); (2, 25); (1, 36)]
```

It could definitely be done in other ways (as well as optimised), but I vaguely remember going with this simply because of it keeping the projected numbers slightly “balanced”.

Finally, we put this together to obtain:

```ocaml
let eval stmt =
    let unb_guess_ids = get_unbounded_guesses stmt in
    if (List.length unb_guess_ids) > 0 then
        let projector = project (List.length unb_guess_ids) in
        let rec helper number = 
            let projection = projector number in
            let guesses = List.combine unb_guess_ids projection in
            let result = eval_stmt [] guesses stmt in
            match result with
                None ->
                    helper (number + 1)
                | Some(_) ->
                    result
        in
        helper 0
    else
        eval_stmt [] [] stmt
```

that will indeed find a solution to the example program from above!

While this is now a working interpreter, it does have its shortcomings when used without any kind of static analysis thrown in. For example, on program:

```
guess x in
reject
```

<!--

the interpreter will enter an infinite loop, despite the fact that the program obviously doesn’t have any solutions. Fixing this will be the topic of the next post.

Oh, and by the way, the name _Kalami_ came from _The Meaning of Liff_ by Douglas Adams and John Lloyd, which I was reading at the time.

-->
