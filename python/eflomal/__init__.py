"""eflomal package"""

from collections import Counter
import logging
from operator import itemgetter
from tempfile import NamedTemporaryFile

from .cython import align, read_text, write_text
import time
#import shutil

logger = logging.getLogger(__name__)

class Aligner:
    """Aligner class"""

    def __init__(self, model=3, score_model=0,
                 n_iterations=None, n_samplers=3,
                 rel_iterations=1.0, null_prior=0.2,
                 source_prefix_len=0, source_suffix_len=0,
                 target_prefix_len=0, target_suffix_len=0):
        self.model = model
        self.score_model = score_model
        self.n_iterations = n_iterations
        self.n_samplers = n_samplers
        self.rel_iterations = rel_iterations
        self.null_prior = null_prior
        self.source_prefix_len = source_prefix_len
        self.source_suffix_len = source_suffix_len
        self.source_lowercase = True
        self.target_prefix_len = target_prefix_len
        self.target_suffix_len = target_suffix_len
        self.target_lowercase = True
        #
        self._preloaded_priors = None
        # Note(preloaded-priors,development): Set to True when developing to
        # ensure consistency between normal and preloaded priors.
        self._assert_preloaded_prior_eq = False

    def preload_priors(self, priors_input):
        """
        Preloads the priors into quick to index structures. Useful in server
        mode, where individual requests typically use a small part of the
        prior words, so iterating the full prior would be wasteful.

        Note that the preprocessing performs the same text transform operations
        that the sentence word transformer would do. So the preprocessed prior
        is already in terms of transformed words, and so is only suitable to
        use with sentence words using the same transformation (which, for the
        same Aligner, is always true).

        """
        t0 = time.time()
        priors = read_priors(priors_input)
        priors_list, hmmf_priors, hmmr_priors, ferf_priors, ferr_priors = priors
        src_tf = TextIndex({}, self.source_prefix_len, self.source_suffix_len,
                           self.source_lowercase)
        trg_tf = TextIndex({}, self.target_prefix_len, self.target_suffix_len,
                           self.target_lowercase)

        priors_tree = {}
        # TODO(NULL): <NULL> is not supported. Could.
        for src_word, trg_word, alpha in priors_list:
            src_word = src_tf.transform(src_word)
            trg_word = trg_tf.transform(trg_word)

            if src_word not in priors_tree:
                priors_tree[src_word] = {}
            trg_tree = priors_tree[src_word]

            trg_tree[trg_word] = trg_tree.get(trg_word, 0.0) + alpha

        ferf_map = {}
        for src_word, fert, alpha in ferf_priors:
            # Note(preloaded-priors,development): for example comment following
            # line to trigger an orig vs preloaded prior difference check.
            src_word = src_tf.transform(src_word)

            if src_word not in ferf_map:
                ferf_map[src_word] = {}
            smap = ferf_map[src_word]
            smap[fert] = smap.get(fert, 0.0) + alpha

        ferr_map = {}
        for trg_word, fert, alpha in ferr_priors:
            trg_word = trg_tf.transform(trg_word)

            if trg_word not in ferr_map:
                ferr_map[trg_word] = {}
            smap = ferr_map[trg_word]
            smap[fert] = smap.get(fert, 0.0) + alpha

        dt = time.time() - t0
        logger.info(f"Prior preprocessing took {dt} seconds")

        preloaded = (priors_tree, ferf_map, ferr_map)

        self._preloaded_priors = (priors, preloaded)

    def prepare_files(self, src_input_file, src_output_file,
                      trg_input_file, trg_output_file,
                      priors_input_file,
                      priors_output_file, orig_priors_output_file=None):
        """Convert text files to formats used by eflomal

        Inputs should be file objects or any iterables over lines. Outputs
        should be file objects.

        """
        src_index, n_src_sents, src_voc_size = to_eflomal_text_file(
            src_input_file, src_output_file,
            self.source_prefix_len, self.source_suffix_len)
        trg_index, n_trg_sents, trg_voc_size = to_eflomal_text_file(
            trg_input_file, trg_output_file,
            self.target_prefix_len, self.target_suffix_len)
        if n_src_sents != n_trg_sents:
            logger.error(
                'number of sentences differ in input files (%d vs %d)',
                n_src_sents, n_trg_sents)
            raise ValueError('Mismatched file sizes')
        logger.info('Prepared %d sentences for alignment', n_src_sents)
        if self._preloaded_priors:
            t0 = time.time()
            (priors, _) = self._preloaded_priors
            preloaded_to_eflomal_priors_file(self._preloaded_priors, src_index,
                                             trg_index, priors_output_file)
            dt = time.time() - t0
            logger.info(f"Prior calculation took {dt} seconds using preloaded")
            if orig_priors_output_file is not None:
                # output normal processing-based priors for comparison
                to_eflomal_priors_file(
                    priors, src_index, trg_index, orig_priors_output_file)
        elif priors_input_file:
            logger.info('Reading lexical priors...')
            priors = read_priors(priors_input_file)
            to_eflomal_priors_file(
                priors, src_index, trg_index, priors_output_file)

    def align(self, src_input, trg_input,
              links_filename_fwd=None, links_filename_rev=None,
              scores_filename_fwd=None, scores_filename_rev=None,
              priors_input=None, quiet=True, use_gdb=False):
        """Run alignment for the input"""
        with NamedTemporaryFile('wb') as srcf, \
             NamedTemporaryFile('wb') as trgf, \
             NamedTemporaryFile('w', encoding='utf-8',
                                delete_on_close=False) as priorsf:

            use_prior = self._preloaded_priors or priors_input
            if self._preloaded_priors and self._assert_preloaded_prior_eq:
                with NamedTemporaryFile('w', encoding='utf-8',
                                        delete_on_close=False) as orig_priorsf:
                    self.prepare_files(
                        src_input, srcf, trg_input, trgf, priors_input,
                        priorsf, orig_priorsf)
                    # Note: opening NamedTemporaryFile-s is safe as long as
                    #  1) happens using context-manager, and 2) delete_on_close
                    #  was set to False, as above.
                    with open(orig_priorsf.name, 'r') as of, \
                         open(priorsf.name, 'r') as f:
                        orig = of.read()
                        pre = f.read()
                        if orig != pre:
                            #shutil.copy(orig_priorsf.name, "/tmp/prior.orig")
                            #shutil.copy(priorsf.name, "/tmp/prior.preloaded")
                            raise Exception("===== ERROR! Preloaded prior leads to differing processed prior! ======")
            else:
                # Write input files for the eflomal binary
                #
                # Note(preloaded-priors): if priors were preloaded, then
                # priors_input is not used at this point (but then likely they
                # are not passed either).
                #
                self.prepare_files(
                    src_input, srcf, trg_input, trgf, priors_input, priorsf)

            # Run wrapper for the eflomal binary
            t0 = time.time()
            align(srcf.name, trgf.name,
                  links_filename_fwd=links_filename_fwd,
                  links_filename_rev=links_filename_rev,
                  statistics_filename=None,
                  scores_filename_fwd=scores_filename_fwd,
                  scores_filename_rev=scores_filename_rev,
                  priors_filename=(priorsf.name if use_prior else None),
                  model=self.model,
                  score_model=self.score_model,
                  n_iterations=self.n_iterations,
                  n_samplers=self.n_samplers,
                  quiet=quiet,
                  rel_iterations=self.rel_iterations,
                  null_prior=self.null_prior,
                  use_gdb=use_gdb)
            dt = time.time() - t0
            logger.info(f"Align call took {dt} seconds")


class TextIndex:
    """
    Word to index mapping with lowercasing and prefix/suffix removal.

    Note that the returned indices are one larger than the indices in the
    passed-in index, due to reserving output index 0 to the <NULL> token.

    """

    def __init__(self, index, prefix_len=0, suffix_len=0, lowercase=True):
        self.index = index
        self.prefix_len = prefix_len
        self.suffix_len = suffix_len
        self.lowercase = lowercase

    def __len__(self):
        return len(self.index)

    def transform(self, word):
        if self.lowercase:
            word = word.lower()
        if self.prefix_len != 0:
            word = word[:self.prefix_len]
        if self.suffix_len != 0:
            word = word[-self.suffix_len:]
        return word

    def __getitem__(self, word):
        word = self.transform(word)
        e = self.index.get(word)
        if e is not None:
            e = e + 1
        return e


def to_eflomal_text_file(sentencefile, outfile, prefix_len=0, suffix_len=0):
    """Write sentences to a file read by eflomal binary

    Arguments:

    sentencefile - input text file object
    outfile - output file object
    prefix_len - prefix length to remove
    suffix_len - suffix length to remove

    Returns TextIndex object.

    """
    sents, index = read_text(sentencefile, True, prefix_len, suffix_len)
    n_sents = len(sents)
    voc_size = len(index)
    write_text(outfile, tuple(sents), voc_size)
    return TextIndex(index, prefix_len, suffix_len), n_sents, voc_size


def sentences_from_joint_file(joint_file, index=None):
    """Yield sentences from joint sentences file"""
    for i, line in enumerate(joint_file):
        fields = line.strip().split(' ||| ')
        if len(fields) != 2:
            logger.error('line %d does not contain a single |||'
                         ' separator, or sentence(s) are empty!',
                         i + 1)
            raise ValueError('Invalid joint input line %s' % line)
        if index is None:
            yield fields[0], fields[1]
        else:
            yield fields[index]


def calculate_priors(src_sentences, trg_sentences,
                     fwd_alignments, rev_alignments,
                     reverse):
    """Calculate priors from alignments

    If `reverse` is True, compute priors for the opposite alignment
    direction.

    Note: stored priors are agnostic of the word transform used during
    alignment, and is in terms of the original sentence words.

    """
    priors = Counter()
    hmmf_priors = Counter()
    hmmr_priors = Counter()
    ferf_priors = Counter()
    ferr_priors = Counter()
    for lineno, (src_sent, trg_sent, fwd_line, rev_line) in enumerate(
            zip(src_sentences, trg_sentences, fwd_alignments, rev_alignments)):
        if lineno % 10000 == 0:
            logger.info('processing line #%d', lineno)
        src_sent = src_sent.strip().split()
        trg_sent = trg_sent.strip().split()
        fwd_links = [tuple(map(int, s.split('-'))) for s in fwd_line.split()]
        rev_links = [tuple(map(int, s.split('-'))) for s in rev_line.split()]
        for i, j in (rev_links if reverse else fwd_links):
            if i >= len(src_sent) or j >= len(trg_sent):
                logger.error('alignment out of bounds in line %d: '
                             '(%d, %d)', lineno + 1, i, j)
                raise ValueError('Invalid input on line %d' % lineno + 1)
            s, t = src_sent[i], trg_sent[j]
            k = (t,s) if rev_alignments else (s,t)
            priors[k] += 1

        last_j = -1
        last_i = -1
        for i, j in sorted(fwd_links, key=itemgetter(1)):
            if j != last_j:
                hmmf_priors[i - last_i] += 1
            last_i = i
            last_j = j
        hmmf_priors[len(src_sent) - last_i] += 1

        last_j = -1
        last_i = -1
        for i, j in sorted(rev_links, key=itemgetter(0)):
            if i != last_i:
                hmmr_priors[j - last_j] += 1
            last_i = i
            last_j = j
        hmmr_priors[len(trg_sent) - last_j] += 1

        fwd_fert = Counter(i for i, j in fwd_links)
        rev_fert = Counter(j for i, j in rev_links)
        for i, fert in fwd_fert.items():
            ferf_priors[(src_sent[i], fert)] += 1
        for j, fert in rev_fert.items():
            ferr_priors[(trg_sent[j], fert)] += 1
    # TODO: confirm EOF in all files

    if reverse:
        return priors, hmmr_priors, hmmf_priors, ferr_priors, ferf_priors
    else:
        return priors, hmmf_priors, hmmr_priors, ferf_priors, ferr_priors


def write_priors(priorsf, priors_list, hmmf_priors, hmmr_priors,
                 ferf_priors, ferr_priors):
    """Write priors to file object"""
    for (src, trg), alpha in sorted(priors_list.items()):
        print('LEX\t%s\t%s\t%g' % (src, trg, alpha), file=priorsf)
    for (src, fert), alpha in sorted(ferf_priors.items()):
        print('FERF\t%s\t%d\t%g' % (src, fert, alpha), file=priorsf)
    for (trg, fert), alpha in sorted(ferr_priors.items()):
        print('FERR\t%s\t%d\t%g' % (trg, fert, alpha), file=priorsf)
    for jump, alpha in sorted(hmmf_priors.items()):
        print('HMMF\t%d\t%g' % (jump, alpha), file=priorsf)
    for jump, alpha in sorted(hmmr_priors.items()):
        print('HMMR\t%d\t%g' % (jump, alpha), file=priorsf)


def read_priors(priors_file):
    """Load priors from file object"""
    priors_list = []    # list of (srcword, trgword, alpha)
    ferf_priors = []    # list of (wordform, alpha)
    ferr_priors = []    # list of (wordform, alpha)
    hmmf_priors = {}    # dict of jump: alpha
    hmmr_priors = {}    # dict of jump: alpha
    # 5 types of lines valid:
    #
    # LEX   srcword     trgword     alpha   | lexical prior
    # HMMF  jump        alpha               | target-side HMM prior
    # HMMR  jump        alpha               | source-side HMM prior
    # FERF  srcword     fert   alpha        | source-side fertility p.
    # FERR  trgword     fert    alpha       | target-side fertility p.
    for i, line in enumerate(priors_file):
        fields = line.rstrip('\n').split('\t')
        try:
            alpha = float(fields[-1])
        except ValueError as err:
            logger.error('priors line %d contains alpha '
                         'value of "%s" which is not numeric',
                         i+1, fields[2])
            raise err
        if fields[0] == 'LEX' and len(fields) == 4:
            priors_list.append((fields[1], fields[2], alpha))
        elif fields[0] == 'HMMF' and len(fields) == 3:
            hmmf_priors[int(fields[1])] = alpha
        elif fields[0] == 'HMMR' and len(fields) == 3:
            hmmr_priors[int(fields[1])] = alpha
        elif fields[0] == 'FERF' and len(fields) == 4:
            ferf_priors.append((fields[1], int(fields[2]), alpha))
        elif fields[0] == 'FERR' and len(fields) == 4:
            ferr_priors.append((fields[1], int(fields[2]), alpha))
        else:
            logger.error('priors line %d is invalid', i + 1)
            raise ValueError('Invalid input on line %d' % i + 1)
    return priors_list, hmmf_priors, hmmr_priors, ferf_priors, ferr_priors


def to_eflomal_priors_file(priors, src_index, trg_index, outfile):
    """Write priors to a file read by eflomal binary

    Arguments:

    priors - tuple of priors (priors_list, hmmf_priors, hmmr_priors,
             ferf_priors, ferr_priors)
    src_index - vocabulary index for source text
    tgt_index - vocabulary index for target text
    outfile - file object for output

    """
    priors_list, hmmf_priors, hmmr_priors, ferf_priors, ferr_priors = priors
    priors_indexed = {}
    for src_word, trg_word, alpha in priors_list:
        if src_word == '<NULL>':
            e = 0
        else:
            e = src_index[src_word]

        if trg_word == '<NULL>':
            f = 0
        else:
            f = trg_index[trg_word]

        if (e is not None) and (f is not None):
            priors_indexed[(e, f)] = priors_indexed.get((e, f), 0.0) \
                + alpha
    ferf_indexed = {}
    for src_word, fert, alpha in ferf_priors:
        e = src_index[src_word]
        if e is not None:
            ferf_indexed[(e, fert)] = \
                ferf_indexed.get((e, fert), 0.0) + alpha
    ferr_indexed = {}
    for trg_word, fert, alpha in ferr_priors:
        f = trg_index[trg_word]
        if f is not None:
            ferr_indexed[(f, fert)] = \
                ferr_indexed.get((f, fert), 0.0) + alpha
    logger.info('%d (of %d) pairs of lexical priors used',
                len(priors_indexed), len(priors_list))
    print('%d %d %d %d %d %d %d' % (
        len(src_index)+1, len(trg_index)+1, len(priors_indexed),
        len(hmmf_priors), len(hmmr_priors),
        len(ferf_indexed), len(ferr_indexed)),
          file=outfile)
    for (e, f), alpha in sorted(priors_indexed.items()):
        print('%d %d %g' % (e, f, alpha), file=outfile)
    for jump, alpha in sorted(hmmf_priors.items()):
        print('%d %g' % (jump, alpha), file=outfile)
    for jump, alpha in sorted(hmmr_priors.items()):
        print('%d %g' % (jump, alpha), file=outfile)
    for (e, fert), alpha in sorted(ferf_indexed.items()):
        print('%d %d %g' % (e, fert, alpha), file=outfile)
    for (f, fert), alpha in sorted(ferr_indexed.items()):
        print('%d %d %g' % (f, fert, alpha), file=outfile)
    outfile.flush()

def preloaded_to_eflomal_priors_file(pp, src_index, trg_index, outfile):
    """Write priors to a file read by eflomal binary

    Arguments:

    priors - tuple of priors (priors_list, hmmf_priors, hmmr_priors,
             ferf_priors, ferr_priors)
    src_index - vocabulary index for source text
    tgt_index - vocabulary index for target text
    outfile - file object for output

    """
    (priors, preloaded_priors) = pp
    priors_list, hmmf_priors, hmmr_priors, ferf_priors, ferr_priors = priors
    (priors_tree, ferf_map, ferr_map) = preloaded_priors

    priors_indexed = {}
    # TODO(NULL): not yet supported.
    for src_word, e in src_index.index.items():
        e = e + 1
        trg_tree = priors_tree.get(src_word)
        if trg_tree is None: continue
        for trg_word, f in trg_index.index.items():
            f = f + 1
            alpha = trg_tree.get(trg_word)
            if alpha is not None:
                priors_indexed[(e, f)] = priors_indexed.get((e, f), 0.0) + alpha

    logger.info('%d (of %d) pairs of lexical priors used',
                len(priors_indexed), len(priors_list))

    ferf_indexed = {}
    for src_word, e in src_index.index.items():
        e = e + 1
        falphas = ferf_map.get(src_word)
        if falphas is None: continue
        for fert, alpha in falphas.items():
            ferf_indexed[(e, fert)] = ferf_indexed.get((e, fert), 0.0) + alpha

    ferr_indexed = {}
    for trg_word, f in trg_index.index.items():
        f = f + 1
        falphas = ferr_map.get(trg_word)
        if falphas is None: continue
        for fert, alpha in falphas.items():
            ferr_indexed[(f, fert)] = ferr_indexed.get((f, fert), 0.0) + alpha

    print('%d %d %d %d %d %d %d' % (
        len(src_index)+1, len(trg_index)+1, len(priors_indexed),
        len(hmmf_priors), len(hmmr_priors),
        len(ferf_indexed), len(ferr_indexed)),
          file=outfile)
    for (e, f), alpha in sorted(priors_indexed.items()):
        print('%d %d %g' % (e, f, alpha), file=outfile)
    for jump, alpha in sorted(hmmf_priors.items()):
        print('%d %g' % (jump, alpha), file=outfile)
    for jump, alpha in sorted(hmmr_priors.items()):
        print('%d %g' % (jump, alpha), file=outfile)
    for (e, fert), alpha in sorted(ferf_indexed.items()):
        print('%d %d %g' % (e, fert, alpha), file=outfile)
    for (f, fert), alpha in sorted(ferr_indexed.items()):
        print('%d %d %g' % (f, fert, alpha), file=outfile)
    outfile.flush()

