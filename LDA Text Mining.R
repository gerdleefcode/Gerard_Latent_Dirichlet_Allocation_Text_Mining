# Divert output to a text file
# sink("Gerard_LDA_Text_Mining_260126.txt")

# cat("\n----------------------------------")
# cat("\nPublisher: Lee Cheuk Man Gerard")
# cat("\nPublication date: 26th January 2026")
# cat("\n----------------------------------\n")

# --------- Section 1: Install required packages and load libraries ----------
pkgs <- c("httr","jsonlite","dplyr","tidyr","stringr","tibble",
          "tidytext","topicmodels","tm","SnowballC","LDAvis","slam", "servr")
to_install <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
if(length(to_install)) install.packages(to_install)
library(httr); library(jsonlite); library(dplyr); library(tidyr); library(stringr)
library(tibble); library(tidytext); library(topicmodels); library(tm); library(SnowballC)
library(LDAvis); library(slam) # R Version 4.5.0

# --------- Section 2: Fetch the Reddit thread JSON (replace username in user_agent) ----------
post_url <- "https://www.reddit.com/r/ArtificialInteligence/comments/1ic1xql/using_ai_at_work/.json"
ua <- "script:lda_example:v0.1 (by /u/your_reddit_username)"
res <- httr::GET(post_url, httr::user_agent(ua))
httr::stop_for_status(res)
raw_json <- httr::content(res, as = "text", encoding = "UTF-8")
j <- jsonlite::fromJSON(raw_json, simplifyVector = FALSE)

# --------- Section 3: Extract Original Post (OP) (title + body) ----------
op <- j[[1]]$data$children[[1]]$data
op_text <- paste0(op$title, "\n", ifelse(is.null(op$selftext), "", op$selftext))

# --------- Section 4: Recursive function to extract comment bodies ----------
extract_bodies <- function(node) {
  out <- character(0)
  if (is.null(node)) return(out)
  
  # collect the comment body if present
  if (!is.null(node$body)) out <- c(out, node$body)
  
  # replies can be "" (character) OR a list with replies$data$children
  replies <- node$replies
  
  # safe check: not NULL and not the empty-string placeholder
  if (!is.null(replies) && !identical(replies, "")) {
    # ensure the structure we expect exists
    if (!is.null(replies$data) && !is.null(replies$data$children)) {
      kids <- replies$data$children
      if (length(kids) > 0) {
        for (k in kids) {
          # skip "more" nodes (they don't have full comment 'data')
          if (!is.null(k$kind) && k$kind == "t1") {
            out <- c(out, extract_bodies(k$data))
          }
        }
      }
    }
  }
  
  out
}

# --------- Section 5: Collect comments (skip "more" nodes) ----------
comments <- character(0)
children <- j[[2]]$data$children
for(ch in children) {
  if(!is.null(ch$kind) && ch$kind == "t1") {
    comments <- c(comments, extract_bodies(ch$data))
  }
}

# --------- Section 6: Build documents dataframe ----------
documents <- c(op_text, comments)
docs_df <- tibble(doc_id = paste0("doc", seq_along(documents)), text = documents)

# --------- Section 7: Simple cleaning + tokenization (tidytext) ----------
data("stop_words")  # tidytext stop words
tokens <- docs_df %>%
  mutate(text = iconv(text, "UTF-8", "UTF-8", sub = "")) %>%
  unnest_tokens(word, text) %>%
  filter(!word %in% stop_words$word) %>%           # remove stopwords
  filter(!str_detect(word, "^\\d+$")) %>%          # drop pure numbers
  mutate(word = wordStem(word)) %>%                # word stemming
  filter(str_length(word) > 2)                     # keep tokens with >2 chars

# --------- Section 8: Build Document-Term Matrix (tm::DocumentTermMatrix) ----------
dtm <- tokens %>%
  count(doc_id, word, sort = TRUE) %>%
  cast_dtm(document = doc_id, term = word, value = n)

# remove empty docs (if any)
row_totals <- slam::row_sums(dtm)
if(any(row_totals == 0)) dtm <- dtm[row_totals > 0, ]

# check size
cat("documents =", nrow(dtm), " terms =", ncol(dtm), "\n")

# --------- Section 9: Fit Latent Dirichlet Allocation (LDA) (topicmodels) ----------
# choose k (number of topics). Must be <= number of documents ideally.
k <- max(2, min(5, nrow(dtm)-1))  # example fallback
lda_model <- LDA(dtm, k = k, method = "Gibbs",
                       control = list(seed = 1234,
                                      burnin = 1000,
                                      iter = 2000,
                                      thin = 200,
                                      alpha = 50/k,
                                      delta = 0.1))

# --------- Section 10: Explore top terms per topic ----------
library(tidytext)
beta <- tidy(lda_model, matrix = "beta")   # word-topic probabilities
top_terms <- beta %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  arrange(topic, -beta)
print(top_terms)

# --------- Section 11: Document-topic probabilities (gamma) & top docs per topic ----------
gamma <- tidy(lda_model, matrix = "gamma")  # document-topic probabilities
top_docs <- gamma %>%
  group_by(topic) %>%
  slice_max(gamma, n = 10) %>%
  arrange(topic, -gamma) %>%
  left_join(docs_df, by = c("document" = "doc_id"))
# print summary of top docs per topic (shows the text)
print(top_docs %>% select(topic, gamma, document, text))

# End of script

# ---------------- Section 12: Save summary outputs to CSV (macOS Desktop) ----------------
out_dir <- file.path(Sys.getenv("HOME"), "Desktop", "lda_output")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# metadata
metadata <- tibble(
  run_time = format(Sys.time(), tz = "", usetz = TRUE),
  documents = nrow(dtm),
  terms = ncol(dtm),
  topics = k
)
write.csv(as.data.frame(metadata), file = file.path(out_dir, "lda_metadata.csv"), row.names = FALSE)

# top terms per topic (the printed summary)
write.csv(as.data.frame(top_terms), file = file.path(out_dir, "lda_top_terms.csv"), row.names = FALSE)

# full beta (word-topic probabilities)
write.csv(as.data.frame(beta), file = file.path(out_dir, "lda_beta.csv"), row.names = FALSE)

# full gamma (document-topic probabilities)
write.csv(as.data.frame(gamma), file = file.path(out_dir, "lda_gamma.csv"), row.names = FALSE)

# top docs per topic (includes text)
write.csv(as.data.frame(top_docs %>% select(topic, gamma, document, text)),
          file = file.path(out_dir, "lda_top_docs.csv"), row.names = FALSE)

# per-document top-topic assignment (most probable topic per document) + original text
doc_top <- gamma %>%
  group_by(document) %>%
  slice_max(gamma, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  left_join(docs_df, by = c("document" = "doc_id"))
write.csv(as.data.frame(doc_top), file = file.path(out_dir, "lda_doc_top_assignment.csv"), row.names = FALSE)

# document term counts (final dtm)
doc_names <- if (!is.null(rownames(dtm))) rownames(dtm) else if (!is.null(dtm$dimnames$Docs)) dtm$dimnames$Docs else paste0("doc", seq_len(nrow(dtm)))
doc_term_counts <- tibble(document = doc_names, term_count = slam::row_sums(dtm))
write.csv(as.data.frame(doc_term_counts), file = file.path(out_dir, "lda_doc_term_counts.csv"), row.names = FALSE)

# full document-term matrix (may be large) — uncomment if desired
dtm_mat <- as.matrix(dtm)
write.csv(dtm_mat, file = file.path(out_dir, "lda_document_term_matrix.csv"), row.names = TRUE)

message("LDA output CSVs written to: ", normalizePath(out_dir))
# ----------------------------------------------------------------

# Stop diverting output and return to the console
# sink()