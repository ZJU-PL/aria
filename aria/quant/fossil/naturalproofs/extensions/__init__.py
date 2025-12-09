"""
Package extending the natural proofs package with finite models extracted from satisfying models.  
The basis for such an extraction being possible is because the models are obtained from the 
satisfaction of quantifier-free formulae. Therefore restricting the obtained to terms of the 
foreground sort appearing in the goal will suffice as a satisfying model for the same goal.  

The extensions package itself will contain various modules for extracting, accessing, and manipulating 
such finite models. Since they are meant to be an explicit representation requiring constant inspection, the models 
are not accessed through an API for basic read/update operations. They are simply wrapped in a class with logging 
attributes.  
"""
