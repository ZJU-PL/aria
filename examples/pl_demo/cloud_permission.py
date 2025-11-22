"""
Permission is a fundamental aspect of any system, and it is especially important in the cloud.

https://github.com/vlab-cs-ucsb/quacky

Quacky quantitatively assesses the (relative) permissiveness of access control policies for the cloud. It
- Translates policies into constraint formulas that conform to the SMT-LIB 2 standard, and
- Counts models satisfying the formulas using the model counting constraint solver ABC.

Quacky supports access control policies written in the following policy languages:

- Amazon Web Services (AWS) Identity and Access Management (IAM)
- Microsoft Azure
- Google Cloud Platform (GCP)

Related work:
- https://github.com/applyfmsec/cloudsec
"""