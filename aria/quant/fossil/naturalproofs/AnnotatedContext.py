# This module defines various annotations over the expressions used in z3py
# This is a hack around z3py's handling of AstRef objects where the ast objects of subterms are not persistent across
# constructions of the ast objects of superterms. Therefore annotations have to be maintained outside.

import z3


class AnnotatedContext:
    """
    Class with annotations about various functions.  
    Current annotations:  
    - alias-annotation: keeps track of an 'alias' for the domain/range sorts of various functions.  
    - Vocabulary-annotation: tracks all the uninterpreted functions and constant declarations (in smt).  
    - Recdef-annotation: tracks all the recursively defined functions and their definitions.  
    - Axiom-annotation: tracks all the axioms.  
    - Variable-annotation: tracks which tracked declarations are 'variables' versus 'constants' in a user-facing sense.  
    """
    def __init__(self):
        self.__alias_annotation__ = dict()
        self.__vocabulary_annotation__ = set()
        self.__recdef_annotation__ = set()
        self.__axiom_annotation__ = set()
        self.__variable_annotation__ = set()

    # Functions to manipulate __alias_annotation__
    def read_alias_annotation(self, funcdeclref):
        """
        Returns the alias annotation given an expression if present in self.  
        :param funcdeclref: z3.FuncDeclRef  
        :return: tuple of naturalproofs.uct.UCTSort objects or None  
        """
        if not isinstance(funcdeclref, z3.FuncDeclRef):
            raise TypeError('FuncDeclRef expected.')
        else:
            key = _alias_annotation_key_repr(funcdeclref)
            signature_annotation = self.__alias_annotation__.get(key, None)
            return signature_annotation

    def is_tracked_alias(self, funcdeclref):
        """
        Returns if the given argument is tracked in the __alias_annotation__  
        :param funcdeclref: z3.FuncDeclRef  
        :return: bool  
        """
        if not isinstance(funcdeclref, z3.FuncDeclRef):
            raise TypeError('FuncDeclRef expected.')
        return _alias_annotation_key_repr(funcdeclref) in self.__alias_annotation__.keys()

    def add_alias_annotation(self, funcdeclref, signature, update=False):
        """
        Adds to the __signature_alias_annotation__ dictionary keyed by a representation of the given
        expression, where the value is the aliased signature of the expression. The expression is
        meant to be a function, and its signature is (**input-sorts, output-sort).Constants are functions with
        only one component in the signature.  
        :param funcdeclref: z3.FuncDeclRef  
        :param signature: tuple of naturalproofs.uct.UCTSort objects  
        :param update: bool (if update is False then previous entries cannot be overwritten)  
        :return: None  
        """
        if not isinstance(funcdeclref, z3.FuncDeclRef):
            raise TypeError('FuncDeclRef Expected.')
        else:
            key = _alias_annotation_key_repr(funcdeclref)
            previous_value = self.read_alias_annotation(funcdeclref)
            if not update and previous_value is not None:
                raise ValueError('Entry already exists. To override provide update=True.')
            if not isinstance(signature, tuple):
                # The expr is a constant and signature was the sort of the expr
                signature = tuple([signature])
            self.__alias_annotation__[key] = signature

    # Functions to manipulate __vocabulary_annotation__
    def get_vocabulary_annotation(self):
        """
        Returns all the uninterpreted functions and constants tracked by self.  
        :return: set of z3.FuncDeclRef objects  
        """
        return self.__vocabulary_annotation__

    def is_tracked_vocabulary(self, funcdeclref):
        """
        Returns if the given argument is tracked in the __vocabulary_annotation__  
        :param funcdeclref: z3.FuncDeclRef  
        :return: bool  
        """
        if not isinstance(funcdeclref, z3.FuncDeclRef):
            raise TypeError('FuncDeclRef expected.')
        return funcdeclref in self.__vocabulary_annotation__

    def add_vocabulary_annotation(self, funcdeclref):
        """
        Adds an annotation to the __vocabulary_annotation__ in self. The annotation is a z3.FuncDeclRef  
        :param funcdeclref: z3.FuncDeclRef object  
        :return: None  
        """
        if not isinstance(funcdeclref, z3.FuncDeclRef):
            raise TypeError('FuncDeclRef expected.')
        self.__vocabulary_annotation__.add(funcdeclref)

    # Functions to manipulate __recdef_annotation__
    def get_recdef_annotation(self):
        """
        Returns all the recursive definitions tracked by self.  
        :return: set of (z3.FuncDeclRef, any, any)  
        """
        return self.__recdef_annotation__

    def add_recdef_annotation(self, annotation):
        """
        Adds an annotation to the __recdef_annotation__ in self. Each recursive definition annotation is a triple. The
        first component of the triple is a z3.FuncDeclRef that is expected to be tracked by __vocabulary_annotation__.
        The second and third components are bound variables and the body of the definition, respectively.  
        :param annotation: (z3.FuncDeclRef, any, any)  
        :return: None  
        """
        self.__recdef_annotation__.add(annotation)

    # Functions to manipulate __axiom_annotation__
    def get_axiom_annotation(self):
        """
        Returns all the axioms tracked by self.  
        :return: set of (any, any)  
        """
        return self.__axiom_annotation__

    def add_axiom_annotation(self, annotation):
        """
        Adds an annotation to the __axiom_annotation__ in self. Each axiom is a pair of bound variables and the body of
        the axiom, respectively.  
        :param annotation: (any, any)  
        :return: None  
        """
        self.__axiom_annotation__.add(annotation)

    # Functions to manipulate __variable_annotation__
    def get_variable_annotation(self):
        """
        Returns all the variables tracked by self.  
        :return: set of z3.ExprRef  
        """
        return self.__variable_annotation__

    def add_variable_annotation(self, annotation):
        """
        Adds an annotation to the __variable_annotation__ in self. Each variable is a z3.ExprRef that is tracked by 
        __alias_annotation__.  
        :param annotation: z3.ExprRef  
        :return: None  
        """
        self.__variable_annotation__.add(annotation)

    def get_recdef(self, recdef_name):
        """
        Returns the recursive definition from the annotated context given the recdef name.
        :param recdef_name: z3.FuncDeclRef
        :return z3.BoolRef
        """
        recs = self.get_recdef_annotation()
        for rec in recs:
            if rec[0] == recdef_name:
                return rec[2]
        return None

# Default annotated context. Only one context needed currently.
default_annctx = AnnotatedContext()


def _alias_annotation_key_repr(astref):
    # Function to convert AstRef objects to representation against which annotations for that object
    # can be stored in the __signature_alias_annotation__ dictionary.
    return astref.__repr__()
