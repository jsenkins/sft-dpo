TEMPLATE = "### Instruction:\n{x}\n\n### Response:\n"
def fmt(x): return TEMPLATE.format(x=x.strip())
