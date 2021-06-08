import cobra
from cobra.test import create_test_model
import numpy
import pandas

import driven
from driven.data_sets import ExpressionProfile
from optlang.symbolics import Zero
from six import iteritems

# installation instructions:
# in a virtual environment (e.g., conda), install cobrapy via pip:
# pip install cobra
#
# Then install the devel branch of my fork of Driven:
# pip install -e git+https://github.com/gregmedlock/driven@devel#egg=driven


# This is a modified version of the standard GIMME algorithm from driven.
# In this version, genes that don't have transcripts detected in the data
# are assigned the max flux penalty. When the original algorithm was
# published, most transcriptomics data was being generated from microarrays
# that use a pre-constructed set of nucleotide probes. If a gene did not have
# a probe on the microarray, there was no way to tell if transcripts were
# present or not, so the authors decided to have zero penalty for those genes.
def gimme_mod(model, expression_profile, cutoff, fraction_of_optimum=0.9,
          condition=0, max_penalty=1):
    r"""
    Modified version of the GIMME algorithm which applies the maximum
    flux penalty to reactions that have no associated transcripts in the
    dataset.
    
    Parameters
    ----------
    model: cobra.Model
        A constraint-based model to perform GIMME on.
    expression_profile: ExpressionProfile
        An expression profile to integrate in the model.
    cutoff: float
        The cutoff value to be defined by the user.
    fraction_of_optimum: float
        The fraction of the Required Metabolic Functionalities.
    condition: str or int, optional (default 0)
        The condition from the expression profile.
        If None, the first condition is used.
    max_penalty: float
        The maximum penalty possible given the users preprocessing
        of transcriptomics data. This penalty will be applied to all
        reactions without any associated transcripts.
    Returns
    -------
    context-specific model: cobra.Model
    solution: cobra.Solution
    Notes
    -----
    The formulation for obtaining the Inconsistency Score is given below:
    minimize: \sum c_i * |v_i|
    s.t.    : Sv = 0
              a_i <= v_i <= b_i
    where   : c_i = {x_cutoff - x_i where x_cutoff > x_i
                     0 otherwise} for all i
    References
    ----------
    .. [1] Becker, S. and Palsson, B. O. (2008).
           Context-specific metabolic networks are consistent with experiments.
           PLoS Computational Biology, 4(5), e1000082.
           doi:10.1371/journal.pcbi.1000082
    """
    with model:
        solution = model.slim_optimize() # returns the flux through the objective
        prob = model.problem # extracts the optimization problem
        rxn_profile = expression_profile.to_reaction_dict(condition, model)
        
        if model.objective_direction == 'max':
            fix_obj_const = prob.Constraint(model.objective.expression,
                                            lb=fraction_of_optimum * solution,
                                            name="RMF")
        else:
            fix_obj_const = prob.Constraint(model.objective.expression,
                                            ub=fraction_of_optimum * solution,
                                            name="RMF")
        model.add_cons_vars(fix_obj_const)

        coefficients = {rxn_id: cutoff - expression
                        for rxn_id, expression in iteritems(rxn_profile)
                        if cutoff > expression}
        
        
        obj_vars = []
        for rxn_id, coefficient in iteritems(coefficients):
            rxn = model.reactions.get_by_id(rxn_id)
            obj_vars.append((rxn.forward_variable, coefficient))
            obj_vars.append((rxn.reverse_variable, coefficient))

        # Add the max penalty to all reactions in the model
        # that are not already in the rxn_profile (e.g., no expression)
        for reaction in model.reactions:
            if reaction.id not in rxn_profile.keys():
                obj_vars.append((reaction.forward_variable,max_penalty))
                obj_vars.append((reaction.reverse_variable,max_penalty))
            
        model.objective = prob.Objective(Zero, sloppy=True, direction="min")
        model.objective.set_linear_coefficients({v: c for v, c in obj_vars})
        sol = model.optimize()
        return model, sol



# create a test model. By default, this is the E. coli core model.
model = create_test_model()

# create a dataframe containing simulated trancript abundances
num_genes = len(model.genes)
num_samples = 10
# driven expects a table of genes (rows) x samples (columns):
transcript_df = pandas.DataFrame(numpy.random.randint(low=0,high=100,size=(num_genes,num_samples)))
transcript_df.index = [gene.id for gene in model.genes]

# transform the simulated counts using the rank-based approach
for sample,subdata in transcript_df.groupby(by=transcript_df.columns,axis=1):
    # within the sample, sort transcripts by counts
    transcripts_ranked = subdata[sample].sort_values()
    # create a vector of ranks from 0 to the length of transcripts
    rank_vec = [x for x in range(0,len(transcripts_ranked))]
    rank_vec = pandas.Series(rank_vec)
    rank_vec.index = transcripts_ranked.index
    # convert the ranks to a percentile
    percentiles = (rank_vec + 1)/len(rank_vec)
    # reorder the percentiles to match the original dataframe
    percentiles = percentiles.reindex(transcript_df.index)
    # multiply the original counts by the percentiles
    transcript_df[sample] = transcript_df[sample].multiply(percentiles)

# Now normalize each transcript by dividing by the max percentile-normalized
# value for that transcript across all samples
transcript_df = transcript_df.div(transcript_df.max(axis=1), axis=0)

# create the expression profile object using Driven
exp_prof = ExpressionProfile(identifiers=transcript_df.index.values,
                             conditions=transcript_df.columns.values,
                             expression=transcript_df.values)

# rename the genes to have a non-numeric character so they don't get evaluated prematurely by driven.
# This is a problem with the AST parser that cobrapy uses to represent gene identifiers.
rename_dict = {gene.id:'gene'+gene.id for gene in model.genes}
cobra.manipulation.modify.rename_genes(model,rename_dict)
model.repair()
# update the GPRs with the 'gene' prefix that we added to the gene IDs
for reaction in model.reactions:
    old_rule = reaction.gene_reaction_rule
    split_by_white = old_rule.split(' ')
    new_rule = ''
    for entry in split_by_white:
        if entry.isdigit():
            new_rule += ' gene'+entry
        else:
            new_rule += ' ' + entry
    reaction.gene_reaction_rule = new_rule


gimme_solutions = {}
with model:
    for sample in transcript_df.columns:
        constrained_model,gimme_solution = gimme_mod(model,
                                                      exp_prof,
                                                      condition=sample,
                                                      cutoff = 1.0,
                                                      fraction_of_optimum = 0.1,
                                                      max_penalty=1.0)
        gimme_solutions[sample] = gimme_solution

# print the fluxes in the solution for the first sample
print(gimme_solutions[transcript_df.columns[0]].fluxes)

