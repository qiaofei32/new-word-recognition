# -* encoding=utf8 *-
import os
import re
import sys
import csv
import pandas
import glob
import math
import time
import jieba
import logging
import argparse

class NewWordRG(object):
	"""
	new word recognition and word frequency count program
	"""
	def __init__(self, corpus_file=None,
				 max_lines_processed=None,
				 MIN_MI=None, MIN_LENTROPY=None,
				 MIN_RENTROPY=None, MIN_TF=None):
		"""
		initializ
		:param corpus_file: The corpus file name or dir name
		"""
		self.MIN_MI = MIN_MI if MIN_MI else 50
		self.MIN_LENTROPY = MIN_LENTROPY if MIN_LENTROPY else 2.0
		self.MIN_RENTROPY = MIN_RENTROPY if MIN_RENTROPY else 2.0
		self.MIN_TF = MIN_TF if MIN_TF else 100
		self.CONTEXT_WINDOW = 5
		self.CONTEXT_PADDING = 1
		self.MSG_ENCODING = "utf8"
		self.max_lines_processed = max_lines_processed
		self.result_dir = ""

		self.corpus_file_list = []
		if corpus_file is None and os.path.isdir("corpus/"):
			self.result_dir = ""
			for file_name in glob.glob("corpus/*"):
				self.corpus_file_list.append(file_name)
		elif os.path.isdir(corpus_file):
			t = corpus_file.replace("\\", "/").split("/")
			t = [i for i in t if i]
			self.result_dir = t[-1]
			for file_name in glob.glob(corpus_file+"/*"):
				self.corpus_file_list.append(file_name)
		elif os.path.isfile(corpus_file):
			self.result_dir = os.path.basename(corpus_file)
			self.corpus_file_list.append(corpus_file)

		logging.basicConfig()
		logger = logging.getLogger()
		logger.setLevel(logging.INFO)
		# DEBUG < INFO < WARNING < ERROR < CRITICAL

		if not self.corpus_file_list:
			logger.error("corpus file not found!")
			sys.exit(1)

		STOP_WORDS_DF = pandas.read_csv(
			filepath_or_buffer="data/stop_words.txt",
			header=None,
			encoding="utf8",
			sep="\t",
			quoting=csv.QUOTE_NONE
		)
		self.STOP_WORDS = STOP_WORDS_DF[0].tolist()
		self.STOP_WORDS_SET = set(self.STOP_WORDS)

		WORD_DICT_DF = pandas.read_csv(
			filepath_or_buffer="data/jieba-dict.txt",
			header=None,
			encoding="utf8",
			sep="\t",
			quoting=csv.QUOTE_NONE
		)
		self.WORD_LIST = WORD_DICT_DF[0].tolist()

		self.ALL_WORDS = self.STOP_WORDS + self.WORD_LIST
		self.ALL_WORDS_SET = set(self.ALL_WORDS)

		RESULT_BASE_DIR = "result/%s/" % self.result_dir
		if not os.path.exists(RESULT_BASE_DIR):
			os.mkdir(RESULT_BASE_DIR)

	def is_eng_num(self, word):
		"""
		the word is english or number
		:param word: unicode word
		:return: True or False
		"""
		return re.sub("[0-9a-zA-Z\-:\.]+", u"", word) == u""

	def gen_char_tf(self):
		"""
		generate single char's term frequncy
		"""
		logging.info("="*60)
		logging.info("[+] gen_char_tf start...")
		TF_FILE_NAME = "result/%s/CHAR_TF.csv" % self.result_dir
		if os.path.exists(TF_FILE_NAME):
			return True
		CHAR_TF = {}
		processed = 0
		for corpus_file in self.corpus_file_list:
			logging.info("="*60)
			logging.info(corpus_file)
			for line in open(corpus_file):
				processed += 1
				if processed % 10000 == 0:
					logging.info(processed)
				if self.max_lines_processed and processed >= self.max_lines_processed:
					break
				line = line.decode("utf8")
				for stop_word in self.STOP_WORDS_SET:
					line = line.replace(stop_word, u"")
				line = re.sub("[0-9a-zA-Z]+", u"", line)
				line = re.sub("\s+", u"", line)
				for char in line:
					CHAR_TF[char] = CHAR_TF.get(char, 0) + 1

		columns = ["Word", "TF"]
		CHAR_TF_LIST = sorted(CHAR_TF.items(), key=lambda d: d[1], reverse=True)
		values = [(w.encode("utf8"), c) for w, c in CHAR_TF_LIST]
		result = pandas.DataFrame(values, columns=columns)
		result.to_csv(TF_FILE_NAME, index=False)

	def gen_dict(self):
		logging.info("="*60)
		logging.info("[+] gen_dict start...")
		WORD_COUNTS = {}
		time_b = time.time()
		processed = 0
		last_time = time.time()
		for corpus_file in self.corpus_file_list:
			f = open(corpus_file)
			for line in f:
				try:
					line = line.decode("utf8")
					line = line.strip()
					line = re.sub("[0-9a-zA-Z]+", u" ", line)
					line = re.sub("\s+", u" ", line)
					for stop_word in self.STOP_WORDS_SET:
						line = line.replace(stop_word, u" ")

					words = list(jieba.cut(line))
					line_length = len(words)

					for index, w in enumerate(words):
						if len(w) >= 2 and not self.is_eng_num(w) and w not in self.ALL_WORDS_SET:
							context = "".join(words[max(index - self.CONTEXT_PADDING, 0): min(index + self.CONTEXT_PADDING + 1, line_length)])
							context = context.strip()
							context_length = len(context)
							for i in range(context_length):
								char_i = context[i]
								if char_i == " ":
									continue

								for j in range(i+2, min(i+self.CONTEXT_WINDOW+1, context_length)):
									char = context[j-1]
									if char == " ":
										break
									word = context[i: j]
									if word in self.ALL_WORDS_SET or word.isdigit():
										continue
									if len(word) <= 1:
										continue

									w_count, prefix_dict, sufix_dict = WORD_COUNTS.get(word, (0, {}, {}))
									w_count = w_count + 1
									if i > 0:
										last_word = context[i - 1]
										prefix_dict[last_word] = prefix_dict.get(last_word, 0) + 1
									else:
										last_word = words[max(index - self.CONTEXT_PADDING - 1, 0)][-1]
										prefix_dict[last_word] = prefix_dict.get(last_word, 0) + 1

									if j < context_length - 1:
										next_word = context[j]
										sufix_dict[next_word] = sufix_dict.get(next_word, 0) + 1
									else:
										next_word = words[min(index + self.CONTEXT_PADDING, line_length - 1)][-1]
										sufix_dict[next_word] = sufix_dict.get(next_word, 0) + 1
									WORD_COUNTS[word] = (w_count, prefix_dict, sufix_dict)

				except KeyboardInterrupt:
					logging.error("KeyboardInterrupt")
					break
				except Exception as e:
					logging.error(e)

				processed += 1
				if processed % 10000 == 0:
					cost_time = time.time() - last_time
					last_time = time.time()
					msg = "No: %d  cost: %ds" %(processed, cost_time)
					logging.info(msg)
				if self.max_lines_processed and processed >= self.max_lines_processed:
					break

		values = []
		columns = ["Word", "TF", "Prefix", "Suffix"]
		if len(WORD_COUNTS) <= 1000 * 1000:
			WORD_COUNT_LIST = sorted(WORD_COUNTS.items(), key=lambda d: d[1][0], reverse=True)
			for k, v in WORD_COUNT_LIST:
				k = k.encode("utf8")
				w_count, prefix_dict, sufix_dict = v
				values.append([k, w_count, str(prefix_dict), str(sufix_dict)])
		else:
			for k, v in WORD_COUNTS.items():
				k = k.encode("utf8")
				w_count, prefix_dict, sufix_dict = v
				values.append([k, w_count, str(prefix_dict), str(sufix_dict)])

		result = pandas.DataFrame(values, columns=columns)
		file_name = "result/%s/WORD_COUNT_DICT.csv" % self.result_dir
		result.to_csv(file_name, index=False)
		time_e = time.time()
		msg = "gen_dict cost: %ds" % (time_e - time_b)
		logging.info(msg)

	def gen_dict_all(self):
		logging.info("="*60)
		logging.info("[+] gen_dict_all start...")
		WORD_COUNTS = {}
		CHAR_TF = {}
		processed = 0
		time_b = time.time()
		for corpus_file in self.corpus_file_list:
			for line in open(corpus_file):
				try:
					processed += 1
					if processed % 1000 == 0:
						logging.info(processed)
					if self.max_lines_processed and processed >= self.max_lines_processed:
						break

					line = line.decode("utf8")
					line = re.sub("\s+", u" ", line)
					for stop_word in self.STOP_WORDS_SET:
						line = line.replace(stop_word, u" ")

					for char in line:
						if char and char != u" " and not self.is_eng_num(char):
							CHAR_TF[char] = CHAR_TF.get(char, 0) + 1

					line_count = len(line)
					for i in range(line_count):
						char_i = line[i]
						if char_i == " ":
							continue
						for j in range(i + 2, min(i + WINDOW + 1, line_count)):
							char = line[j - 1]
							if char == " ":
								break
							word = line[i: j]
							if len(word) <= 1 or self.is_eng_num(word):
								continue

							w_count, prefix_dict, sufix_dict = WORD_COUNTS.get(word, (0, {}, {}))
							w_count = w_count + 1
							if i > 0:
								last_word = line[i - 1].strip()
								if last_word:
									prefix_dict[last_word] = prefix_dict.get(last_word, 0) + 1
							if j <= line_count - 1:
								next_word = line[j].strip()
								if next_word:
									sufix_dict[next_word] = sufix_dict.get(next_word, 0) + 1
							WORD_COUNTS[word] = (w_count, prefix_dict, sufix_dict)
				except KeyboardInterrupt:
					logging.error("gen_dict: KeyboardInterrupt!")
					break
				except Exception as e:
					logging.error(str(e))

		values = []
		columns = ["Word", "TF", "Prefix", "Suffix"]
		DICT_FILE_NAME = "result/%s/WORD_COUNT_DICT_ALL.csv" % self.result_dir
		if len(WORD_COUNTS) <= 1000 * 1000:
			WORD_COUNT_LIST = sorted(WORD_COUNTS.items(), key=lambda d: d[1][0], reverse=True)
			for k, v in WORD_COUNT_LIST:
				k = k.encode("utf8")
				w_count, prefix_dict, sufix_dict = v
				values.append([k, w_count, str(prefix_dict), str(sufix_dict)])
		else:
			for k, v in WORD_COUNTS.items():
				k = k.encode("utf8")
				w_count, prefix_dict, sufix_dict = v
				values.append([k, w_count, str(prefix_dict), str(sufix_dict)])

		result = pandas.DataFrame(values, columns=columns)
		result.to_csv(DICT_FILE_NAME, index=False)

		TF_FILE_NAME = "result/%s/CHAR_TF_ALL.csv" % self.result_dir
		columns = ["Word", "TF"]
		char_tf_list = sorted(CHAR_TF.items(), key=lambda d: d[1], reverse=True)
		values = [(w.encode("utf8"), c) for w, c in char_tf_list]
		result = pandas.DataFrame(values, columns=columns)
		result.to_csv(TF_FILE_NAME, index=False)

		time_e = time.time()
		msg = "cost: %ds" % (time_e - time_b)
		logging.info(msg)

	def calc_mi_entropy(self, word,
						CHAR_TF=None,
						WORD_COUNTS=None,
						word_all_count=None,
						char_all_count=None):
		"""
		Calculate MI and entropy of a word
		"""
		TF_FILE_NAME = "data/CHAR_TF.csv"
		MERGE_FILE = "data/WORD_COUNT_DICT.csv"

		if CHAR_TF is None:
			CHAR_TF = pandas.read_csv(TF_FILE_NAME, encoding="utf8", header=0, sep=",", na_filter=False, engine="c")
		if WORD_COUNTS is None:
			WORD_COUNTS = pandas.read_csv(MERGE_FILE, encoding="utf8", header=0, sep=",", na_filter=False, engine="c")
		if char_all_count is None:
			char_all_count = CHAR_TF.TF.sum()
		if word_all_count is None:
			word_all_count = WORD_COUNTS.TF.sum()

		df_word = WORD_COUNTS[WORD_COUNTS.Word == word]
		tf_word = df_word.TF.tolist()[0]

		MI_list = []
		for i in range(1, len(word)):
			part_00 = "".join(word[:i])
			part_01 = "".join(word[i:])

			if len(part_00) >= 2:
				df_00 = WORD_COUNTS[WORD_COUNTS.Word == part_00]
				tf_00 = df_00.TF.tolist()[0]
				p_tf_00 = 1.0 * tf_00 / word_all_count

			else:
				df_00 = CHAR_TF[CHAR_TF.Word == part_00]
				tf_00 = df_00.TF.tolist()[0]
				p_tf_00 = 1.0 * tf_00 / char_all_count

			if len(part_01) >= 2:
				df_01 = WORD_COUNTS[WORD_COUNTS.Word == part_01]
				tf_01 = df_01.TF.tolist()[0]
				p_tf_01 = 1.0 * tf_01 / word_all_count
			else:
				df_01 = CHAR_TF[CHAR_TF.Word == part_01]
				tf_01 = df_01.TF.tolist()[0]
				p_tf_01 = 1.0 * tf_01 / char_all_count

			p_all = 1.0 * tf_word / word_all_count
			MI_i = p_all / (p_tf_00 * p_tf_01)
			MI_list.append(MI_i)

		MI = min(MI_list)
		#=======================================
		prefix_dict = df_word.Prefix.tolist()[0]
		suffix_dict = df_word.Suffix.tolist()[0]
		prefix_dict = eval(prefix_dict)
		suffix_dict = eval(suffix_dict)
		# prefix_dict[u" "] = 0
		# suffix_dict[u" "] = 0

		entropy_left = 0
		N = sum(prefix_dict.values()) * 1.0
		N = max(N, 1)
		for w, c in prefix_dict.items():
			p = c / N
			# print p
			if p > 0.0:
				entropy_left += (-p)*math.log(p)

		entropy_rignt = 0
		N = sum(suffix_dict.values()) * 1.0
		N = max(N, 1)
		for w, c in suffix_dict.items():
			p = c / N
			if p > 0.0:
				entropy_rignt += (-p)*math.log(p)

		# print MI
		# print entropy_left
		# print entropy_rignt
		return MI, entropy_left, entropy_rignt

	def get_words(self):
		TF_FILE_NAME = "result/%s/CHAR_TF.csv" % self.result_dir
		if not os.path.exists(TF_FILE_NAME):
			self.gen_char_tf()

		MERGE_FILE = "result/%s/WORD_COUNT_DICT.csv" % self.result_dir
		if not os.path.exists(MERGE_FILE):
			self.gen_dict()

		logging.info("="*60)
		logging.info("[+] get_words start...")
		CHAR_TF = pandas.read_csv(TF_FILE_NAME, encoding="utf8", header=0, sep=",", na_filter=False, engine="c")
		WORD_COUNTS = pandas.read_csv(MERGE_FILE, encoding="utf8", header=0, sep=",", na_filter=False, engine="c")

		word_all_count = WORD_COUNTS.TF.sum()
		char_all_count = CHAR_TF.TF.sum()

		WORDS = WORD_COUNTS.Word.tolist()
		TFS = WORD_COUNTS.TF.tolist()
		NUM = len(TFS)

		values = []
		new_word_values = []
		for index, word in enumerate(WORDS):
			tf = TFS[index]
			if tf < self.MIN_TF:
				continue
			try:
				MI, entropy_left, entropy_rignt = self.calc_mi_entropy(
					word=word,
					CHAR_TF=CHAR_TF,
					WORD_COUNTS=WORD_COUNTS,
					word_all_count=word_all_count,
					char_all_count=char_all_count
				)

				str_output = "%5d/%d  %s  %s  %s  %s  %s" %(
					index,
					NUM,
					"{:6d}".format(tf)[:6],
					"{:.4f}".format(MI)[:6],
					"{:.4f}".format(entropy_left)[:6],
					"{:.4f}".format(entropy_rignt)[:6],
					word.encode(self.MSG_ENCODING)
				)

				if MI >= self.MIN_MI and (entropy_left >= self.MIN_LENTROPY and entropy_rignt >= self.MIN_RENTROPY):
					logging.info(str_output + " +++++++++++")
					new_word_values.append([word.encode("utf8"), tf, MI, entropy_left, entropy_rignt])
				else:
					logging.info(str_output)
				values.append([word.encode("utf8"), tf, MI, entropy_left, entropy_rignt])
			except KeyboardInterrupt:
				logging.error("get_words: KeyboardInterrupt!")
				break
			except Exception as e:
				# logging.error(str(e))
				pass

		columns = ["Word", "TF", "MI", "LE", "RE"]
		new_word_result = pandas.DataFrame(new_word_values, columns=columns)
		NEW_WORD_FILE_NAME = "result/%s/NEW-WORD.csv" % self.result_dir
		new_word_result.to_csv(NEW_WORD_FILE_NAME, index=False, encoding="utf8")

		result = pandas.DataFrame(values, columns=columns)
		WORD_TF_MI_ENTROPY_FILE_NAME = "result/%s/WORD-TF-MI-ENTROPY.csv" % self.result_dir
		result.to_csv(WORD_TF_MI_ENTROPY_FILE_NAME, index=False, encoding="utf8")


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--method", required=True, help="method to call: gen_dict/gen_char_tf/get_words")
	parser.add_argument("-i", "--input", required=False, help="corpus file path or dir name")
	parser.add_argument("-n", "--lines", required=False, help="max lines to process", type=int)
	parser.add_argument("-p", "--mi", required=False, help="min MI", type=float)
	parser.add_argument("-l", "--lentropy", required=False, help="min left entropy", type=float)
	parser.add_argument("-r", "--rentropy", required=False, help="min right entropy", type=float)
	parser.add_argument("-t", "--tf", required=False, help="min TF", type=float)
	args = parser.parse_args()
	# print args
	method = args.method
	corpus = args.input
	lines = args.lines
	lentropy = args.lentropy
	rentropy = args.rentropy
	mi = args.mi
	tf = args.tf

	# print globals()
	M = globals()['NewWordRG'](
		corpus_file=corpus,
		max_lines_processed=lines,
		MIN_MI=mi,
		MIN_LENTROPY=lentropy,
		MIN_RENTROPY=rentropy,
		MIN_TF=tf
	)
	func = getattr(M, method)
	func()
