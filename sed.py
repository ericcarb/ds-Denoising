'''
README

A modification of single_engine_demo.ipynb

Benchmarks all files declared in ids_names
Generates a few text files:
	errors.txt					any errors that are caught during execution have their messages written here
	error_ids.txt				the ids that were being processed when the error occured
	words_dfs.txt				the words dataframes containing the engine predictions, compiled into a text file
	performance_measures.txt	output of TranscriptionBenchmark.compute_performance_measures
	lattice_details.txt			output of TranscriptionBenchmark.get_lattice_details
'''

from ecosystem.benchmark.transcription import TranscriptionBenchmark
from ecosystem.gql_client import GQLClient
from ecosystem.job import Job, JobManager
from ecosystem.veritone_engines import *
import pandas as pd

# Local imports
demo_config = demo_config = {
    # If this is present, it is used to find ground truth.  Otherwise, a DB lookup is used.
    'ground_truth_local_dir': '/home/eric.carb/GT/', #'/Users/ericcarb/gts/',

    # Ground truth source
    'ground_truth_source': 'voicebase-human',

    # Ground truth temp file
    'ground_truth_temp_file': '/tmp/gt.ttml',

    # Ground truth scrubbed file
    'ground_truth_scrubbed_file': '/tmp/gt.scrubbed',

    # Utterance marker for SCLITE
    'utterance_id': ' (trans_0001)',

    # Path to SCLITE executable
    'sclite': '/home/craths/sctk-2.4.10/bin/sclite',
}

# Init GraphQL client with your token
token = 'jtdscience:APSSnwhnZAIqS3CIB1eXYojVatXMriPj3BQn9FjD2wBd3pF04mdj1hGQ46tzP9ER'
client = GQLClient(token)


# Select Recordings to process
ids_names = [
('01234567', 'filename')
]

# Specify Engine to use

test_engine = Engine('Capio', 
                      'English (US) Broadcast', 
                      'fe12c2cf-15f7-4267-88fb-21f7b820c6d7')

engine_speechmatics = Engine('Speechmatics', 
                             'Supernova-English (USA)',
                             'transcribe-speechmatics-container-en-us')

engine_kaldi = Engine('Kaldi',
                      'Conductor Engine Kaldi',
                      '6ad5c1c4-1c7f-4097-99a4-e48fbc9e719b')

test_engine = engine_speechmatics

test_job = Job(test_engine)
all_jobs = [test_job]


# Create a manager to execute the jobs
jobManager = JobManager(client, all_jobs)

# Setup transcription benchmark
transcription_benchmark = TranscriptionBenchmark(client, demo_config)

# A Pandas data frame to accumulate results for all recordings
all_words_df = None
words_df = None

# Iterate over all recordings to run engines and align results
for recording_id in ids_names:
    recording_name = recording_id[1]
    recording_id = recording_id[0]
    print("START Processing recording ID: %s" % recording_id)
    try:
        jobManager.execute(recording_id)
    except Exception as e:
        print('jobManager.execute({}) raised an EXCEPTION'.format(recording_id))
        with open('errors.txt', 'a') as outf:
            outf.write(str(e))
            outf.write('\n')
        with open('error_ids.txt', 'a') as outf:
            outf.write(str(recording_id))
            outf.write('\n')
        continue


    try:
        # Align the transcripts with ground truth
        words_df = transcription_benchmark.align_lattice_votes(recording_id, all_jobs)
    except Exception as e:
        print('transcription_benchmark.align_lattice_votes({}, all_jobs) raised an EXCEPTION'.format(recording_id))
        with open('errors.txt', 'a') as outf:
            outf.write(str(e))
            outf.write('\n')
        with open('error_ids.txt', 'a') as outf:
            outf.write(str(recording_id))
            outf.write('\n')
        continue

    print('\n###\n', words_df.to_csv(), '\n', sep='')

    with open('words_dfs.txt', 'a') as outf:
        outf.write('\n#{}#{}#\n'.format(recording_name, recording_id))
        outf.write(words_df.to_csv())
        outf.write('\n#\n')

    # Add results for this recording to the Pandas data frame for all words
    if all_words_df is None:
        all_words_df = words_df
    else:
        all_words_df = pd.concat([all_words_df, words_df], axis=0)
    print("END Processing recording ID: %s" % recording_id)


# Display performance measures
perf_measures = transcription_benchmark.compute_performance_measures(all_words_df, all_jobs)


# Show details of word alignments
lattice_details = transcription_benchmark.get_lattice_details(all_words_df, all_jobs)


with open('performance_measures.txt', 'w+') as outf:
    outf.write(repr(perf_measures))
    outf.write('\n')
with open('lattice_details.txt', 'w+') as outf:
    outf.write(lattice_details.to_csv())
    outf.write('\n')

