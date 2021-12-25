'''Homework 1 Python Questions

This is an individual homework
Implement the following functions.

Do not add any more import lines to this file than the ones
already here without asking for permission on Piazza.
Use the regular expression tools built into Python; do NOT use bash.
'''

import re

def check_for_foo_or_bar(text):
  '''Checks whether the input string meets the following condition.

   The string must have both the word 'foo' and the word 'bar' in it,
   whitespace- or punctuation-delimited from other words.
   (not, e.g., words like 'foobar' or 'bart' that merely contain
    the word 'bar');

   See the Python regular expression documentation:
   https://docs.python.org/3.4/library/re.html#match-objects

   Return:
     True if the condition is met, false otherwise.
  '''
  for i in text:
    if not i.isalpha() and not i.isdigit():
      text=text.replace(i,' ')
  text=' '.join([i for i in text.split(' ') if i!='']) # word with single whitespace
  # print(text)

  p1=re.compile('foo(?:(?:\s\w+)+|(?:\w+\s\w+)+|(?:\w+\s)+|(?:\s)+)+bar')
                    # foo xbar    foox xbar      foox bar   foo bar
  p2=re.compile('bar(?:(?:\s\w+)+|(?:\w+\s\w+)+|(?:\w+\s)+|(?:\s)+)+foo')
  exist=p1.search(text) or p2.search(text)

  return bool(exist)

# print(check_for_foo_or_bar('dofie s7& fi foo barfoo sfho,& dsfhoi sfio bar uisu'))
# print(check_for_foo_or_bar('foobar'))

def replace_rgb(text):
  '''Replaces all RGB or hex colors with the word 'COLOR'
   
   Possible formats for a color string:
   #0f0
   #0b013a
   #37EfaA
   rgb(1, 1, 1)
   rgb(255,19,32)
   rgb(00,01, 18)
   rgb(0.1, 0.5,1.0)

   There is no need to try to recognize rgba or other formats not listed 
   above. There is also no need to validate the ranges of the rgb values.

   However, you should make sure all numbers are indeed valid numbers.
   For example, '#xyzxyz' should return false as these are not valid hex digits.
   Similarly, 'rgb(c00l, 255, 255)' should return false.

   Only replace matching colors which are at the beginning or end of the line,
   or are space separated from the text around them. For example, due to the 
   trailing period:

   'I like rgb(1, 2, 3) and rgb(2, 3, 4).' becomes 'I like COLOR and rgb(2, 3, 4).'

   # See the Python regular expression documentation:
   https://docs.python.org/3.4/library/re.html#re.sub

   Returns:
     The text with all RGB or hex colors replaces with the word 'COLOR'
  '''
  # pattern = "rgb([0-9](?:(?:,)+|(?:,\s)+)[0-9](?:(?:,)+|(?:,\s)+)[0-9])"

  # number : \d+\.?\d*
  # separator(,\s or ,) : (?:(?:,)+|(?:,\s)+)
  rgbpattern = "(rgb)?\(\d+\.?\d*(?:(?:,)+|(?:,\s)+)\d+\.?\d*(?:(?:,)+|(?:,\s)+)\d+\.?\d*\)"
  # hexpattern = "\#\d+[A-F]+[a-f]+"
  # hexpattern = "\#(?=\d+)(?=.*[A-F])(?=.*[a-f])"
  hexpattern = "\#([0-9a-fA-F]){1,6}"
  
  patterns=[rgbpattern,hexpattern]
  # pattern = "\s(rgb)?\(\d+\.?\d*(?:(?:,)+|(?:,\s)+)\d+\.?\d*(?:(?:,)+|(?:,\s)+)\d+\.?\d*\)\s"
  def multiple_sub(text,pattern,substr):
    while True:
      new = re.sub(pattern, substr, text)
      if new==text:
        return new
      text=new


  for p in patterns:
    pattern = "\s"+p+"\s"
    text = multiple_sub(text,pattern,substr=" COLOR ")

    # pattern = "^"+p+"$"
    # text = multiple_sub(text,pattern,substr="COLOR")
    pattern = "^"+p+"\s"
    text = multiple_sub(text,pattern,substr="COLOR ")
    
    pattern = "\n"+p
    text = multiple_sub(text,pattern,substr="\nCOLOR")
    
    pattern = p+"\n"
    text = multiple_sub(text,pattern,substr="COLOR\n")
    
    pattern = "\s"+p+"$"
    text = multiple_sub(text,pattern,substr=" COLOR")
    pattern = "^"+p+"$"
    text = multiple_sub(text,pattern,substr="COLOR")

  
  # pattern = "(rgb)?\(\d+\.?\d*(?:(?:,)+|(?:,\s)+)\d+\.?\d*(?:(?:,)+|(?:,\s)+)\d+\.?\d*\)$"
  # text = re.sub(pattern, "COLOR", text)

  return text
# print(replace_rgb('#123rgb(134, 1, 1)'))
# print(replace_rgb('rrrrgb(1,1,1)'))
# print(replace_rgb(' #0000000 '))
# print(replace_rgb('#000000fffff'))
# print(replace_rgb('I like #xyzxyz and #37EfaA\nI like rgb(255,19,32) and rgb(2, 3, 4).'))
# print(replace_rgb('I like #0f0 #0b013a #37EfaA rgb(1, 1, 1) rgb(255,19,32) rgb(00,01,18)\nrgb(0.1, 0.5,1.0)'))


def edit_distance(str1, str2):
  '''Computes the minimum edit distance between the two strings.

  Use a cost of 1 for all operations.

  See Section 2.4 in Jurafsky and Martin for algorithm details.
  Do NOT use recursion.

  Returns:
    An integer representing the string edit distance
    between str1 and str2
  '''
  if not str1 and not str2:
    return 0
  if not str1:
    return max(len(str2),0)
  if not str2:
    return max(len(str1),0)
  n = len(str1)
  m = len(str2)
  # if m==0 or n==0:

  D = [[0] * (m + 1) for i in range(n + 1)]
  
  for i in range(1,n+1):
    D[i][0] = D[i-1][0] + 1
  for j in range(1,m+1):
    D[0][j] = D[0][j-1] + 1
  # print(D)

  for i in range(1,n+1):
    for j in range(1,m+1):
      # if i < len(str1) and j < len(str2) and str1[i] == str2[j]:
      if str1[i-1] == str2[j-1]:
        sub_cost = 0
      else:
        sub_cost = 1
      D[i][j] = min(D[i-1][j] + 1,
                  D[i-1][j-1] + sub_cost,
                  D[i][j-1] + 1)
  return D[n][m]

# e=edit_distance('','sdfo')




def wine_text_processing(wine_file_path, stopwords_file_path):
  '''Process the two files to answer the following questions and output results to stdout.

  1. What is the distribution over star ratings?
  2. What are the 10 most common words used across all of the reviews, and how many times
     is each used?
  3. How many times does the word 'a' appear?
  4. How many times does the word 'fruit' appear?
  5. How many times does the word 'mineral' appear?
  6. Common words (like 'a') are not as interesting as uncommon words (like 'mineral').
     In natural language processing, we call these common words "stop words" and often
     remove them before we process text. stopwords.txt gives you a list of some very
     common words. Remove these stopwords from your reviews. Also, try converting all the
     words to lower case (since we probably don't want to count 'fruit' and 'Fruit' as two
     different words). Now what are the 10 most common words across all of the reviews,
     and how many times is each used?
  7. You should continue to use the preprocessed reviews for the following questions
     (lower-cased, no stopwords).  What are the 10 most used words among the 5 star
     reviews, and how many times is each used? 
  8. What are the 10 most used words among the 1 star reviews, and how many times is
     each used? 
  9. Gather two sets of reviews: 1) Those that use the word "red" and 2) those that use the word
     "white". What are the 10 most frequent words in the "red" reviews which do NOT appear in the
     "white" reviews?
  10. What are the 10 most frequent words in the "white" reviews which do NOT appear in the "red"
      reviews?

  No return value.
  '''
  def get_word_dict(data,ques='2',stopwords=None):
    word_dict = {}
    for line in data:
      if ques!='2':
        line=line.lower()
      comment = line.strip().split('\t')[0]
      rate = line.strip().split('\t')[1]
      if ques=='7' and len(rate)!=5:
        continue
      if ques=='8' and len(rate)!=1:
        continue
      for c in comment.split():
        if ques!='2' and c in stopwords:
          continue
        if c not in word_dict: 
          word_dict[c] = 1
        else:
          word_dict[c] += 1
    return word_dict

  with open(wine_file_path,encoding="utf8", errors='ignore') as f:
    data=f.readlines()

  # 1. distribution over star ratings
  star_dict = {}
  for line in data:
    rate = line.strip().split('\t')[1]
    if rate not in star_dict:
      star_dict[rate] = 1
    else:
      star_dict[rate] += 1
  for k in sorted(star_dict,reverse=True):
    print(k+'\t'+str(star_dict[k]))
  print('')

  # 2. 10 most common words
  word_dict = get_word_dict(data)
  sorted_word = sorted(word_dict.items(),key=lambda x:x[1],reverse=True)
  sorted_word = sorted(sorted_word, key=lambda t: (-t[1], t[0]))
  for w in sorted_word[:10]:
    print(w[0]+'\t'+str(w[1]))
  print('')
  
  # 3. How many times does the word 'a' appear?
  print(word_dict['a'])
  print('')

  # 4. How many times does the word 'fruit' appear?
  print(word_dict['fruit'])
  print('')

  # 5. How many times does the word 'mineral' appear?
  print(word_dict['mineral'])
  print('')

  # 6. 10 most common words w/o stop words
  with open(stopwords_file_path) as f:
    stopwords=f.readlines()
  stopwords = [w.strip() for w in stopwords]
  word_dict = get_word_dict(data,ques='6',stopwords=stopwords)
  sorted_word = sorted(word_dict.items(),key=lambda x:x[1],reverse=True)
  sorted_word = sorted(sorted_word, key=lambda t: (-t[1], t[0]))
  for w in sorted_word[:10]:
    print(w[0]+'\t'+str(w[1]))
  print('')

  # 7. 10 most used words among the 5 star reviews
  word_dict = get_word_dict(data,ques='7',stopwords=stopwords)
  sorted_word = sorted(word_dict.items(),key=lambda x:x[1],reverse=True)
  sorted_word = sorted(sorted_word, key=lambda t: (-t[1], t[0]))
  for w in sorted_word[:10]:
    print(w[0]+'\t'+str(w[1]))
  print('')

  # 8.  10 most used words among the 1 star reviews
  word_dict = get_word_dict(data,ques='8',stopwords=stopwords)
  sorted_word = sorted(word_dict.items(),key=lambda x:x[1],reverse=True)
  sorted_word = sorted(sorted_word, key=lambda t: (-t[1], t[0]))
  for w in sorted_word[:10]:
    print(w[0]+'\t'+str(w[1]))
  print('')

  def get_red_white_dict(a,b,data,stopwords):
    # use word a , does not appear in word-b-reviews
    a_dict,b_dict={},{}
    for line in data:
      comment = line.lower().strip().split('\t')[0]
      comment = [w for w in comment.split() if w not in stopwords]
      if b in comment:
        for c in comment:
          if c not in b_dict: 
            b_dict[c] = 1
          else:
            b_dict[c] += 1
      elif a in comment:
        for c in comment:
          if c not in a_dict: 
            a_dict[c] = 1
          else:
            a_dict[c] += 1
    for k in b_dict.keys(): 
      if k in a_dict: 
        del a_dict[k]
    return a_dict

  # 9. Gather two sets of reviews: 1) Those that use the word "red" and 2) those that use the word
  #    "white". What are the 10 most frequent words in the "red" reviews which do NOT appear in the
  #    "white" reviews?
  # print('!!! 9')
  word_dict = get_red_white_dict('red','white',data,stopwords)
  sorted_word = sorted(word_dict.items(),key=lambda x:x[1],reverse=True)
  sorted_word = sorted(sorted_word, key=lambda t: (-t[1], t[0]))
  for w in sorted_word[:10]:
    print(w[0]+'\t'+str(w[1]))
  print('')



  # 10. What are the 10 most frequent words in the "white" reviews which do NOT appear in the "red"
  #     reviews?
  word_dict = get_red_white_dict('white','red',data,stopwords)
  sorted_word = sorted(word_dict.items(),key=lambda x:x[1],reverse=True)
  sorted_word = sorted(sorted_word, key=lambda t: (-t[1], t[0]))
  for w in sorted_word[:10]:
    print(w[0]+'\t'+str(w[1]))
  print('')

# wine_text_processing('data/wine.txt','data/stopwords.txt')


