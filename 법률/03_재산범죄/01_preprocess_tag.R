#theft
eda_theft <- data[data$tag_crime_group_manual == "theft",]
eda_theft <- eda_theft[eda_theft$guilty == "Guilty",]
eda_theft <- eda_theft[eda_theft$punishment %in% c("suspended", "imprisonment"),]

#theft는 모두 상습누범절도 
#theft는 29건밖에 되지 않아 모두 직접 읽고 태깅했습니다
month <- c(36, 18, 8, NA, 12, 24, NA, 60, 24, NA, 7, 8, NA, NA, 10, NA, NA, 60, 48, NA, NA, 12, 12, NA, 48, 6, NA, 8, NA)
demage <- c(30000000, 1260000, 46000000, 6000, 569600000, 13307000, NA, 111360900, 32000000, NA, NA, 5200, NA, NA, 260000, NA, NA, 1637083700, 8404000, NA, NA, NA, NA, 1637083700, 8404000, 189800, NA, 244200, NA)
repeated <- c(TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE)
refelction <- c(TRUE, TRUE, TRUE, NA, TRUE, TRUE, NA, NA, TRUE, NA, TRUE, TRUE, NA, NA, TRUE, NA, NA, NA, TRUE, NA, NA, TRUE, TRUE, NA, NA, TRUE, NA, TRUE, NA)
conceal <- c(FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE)
settlement <- c(TRUE, NA, FALSE, NA, FALSE, NA, NA, NA, TRUE, NA, FALSE, FALSE, NA, NA, FALSE, NA, NA, FALSE, TRUE, NA, NA, TRUE, NA, FALSE, TRUE, TRUE, NA, NA, NA)
restitution <- c(TRUE, NA, TRUE, NA, FALSE, NA, NA, NA, TRUE, NA, NA, NA, NA, NA, FALSE, NA, NA, TRUE, NA, NA, NA, NA, NA, TRUE, NA, TRUE, NA, TRUE, NA)

eda_theft$month <- month
eda_theft$demage <- damage
eda_theft$repeated <- recidivism
eda_theft$reflection <- confession
eda_theft$conceal <- conceal
eda_theft$settlement <- settlement
eda_theft$restitution <- restitution

eda_theft$repeated <- as.integer(replace(eda_theft$repeated, is.na(eda_theft$repeated), FALSE))
eda_theft$reflection <- as.integer(replace(eda_theft$reflection, is.na(eda_theft$reflection), FALSE))
eda_theft$conceal <- as.integer(replace(eda_theft$conceal, is.na(eda_theft$conceal), FALSE))
eda_theft$settlement <- as.integer(replace(eda_theft$settlement, is.na(eda_theft$settlement), FALSE))
eda_theft$restitution<- as.integer(replace(eda_theft$restitution, is.na(eda_theft$restitution), FALSE))

use_theft <- data.frame(
  lawyer = eda_theft$lawyer,
  punishment = eda_theft$punishment,
  month = eda_theft$month,
  demage = eda_theft$demage,
  repeated = eda_theft$repeated,
  reflection = eda_theft$reflection,
  conceal = eda_theft$conceal,
  settlement = eda_theft$settlement,
  restitution = eda_theft$restitution )

use_theft$crime = "theft"
use_theft$outcome = ifelse(eda_theft$punishment == "suspended",1,0)
write.csv(use_theft, "Theft")


#breach
eda_breach <- data[data$tag_crime_group_manual %in% c("embezzlement_breach", "bribery_corruption"),]
eda_breach <- eda_breach[eda_breach$guilty == "Guilty",]
eda_breach <- eda_breach[eda_breach$punishment %in% c("both", "imprisonment", "suspended"),]

#breach_month
sentence_to_year <- function(x) {
  year <- 0
  month <- 0
  
  if (grepl("[0-9]+년", x)) {
    year <- as.numeric(sub(".*?([0-9]+)년.*", "\\1", x))}
  
  if (grepl("[0-9]+개월", x)) {
    month <- as.numeric(sub(".*?([0-9]+)개월.*", "\\1", x))}
  
  else if (grepl("[0-9]+월", x)) {
    month <- as.numeric(sub(".*?([0-9]+)월.*", "\\1", x))}
  
  year*12 + month}

year <- c()
for (i in 1:nrow(eda_breach)) {
  x <- Guilty_FRAUD$full_text[i]
  x <- gsub("\n", "", x)
  x <- unlist(strsplit(x, split = " "))
  
  if (any(x == "징역", na.rm = TRUE)) {
    ind <- which(x == "징역")[1]
    year[i] <- sentence_to_year(paste0(x[ind:(ind+2)], collapse = ""))}
  
  else {year[i] <- NA}}

eda_breach$month <- year

#breach$demage
extract_damage <- function(text) {
  if (is.na(text) | text == "") return(NA)
  text <- gsub("\n", " ", text)
  sentences <- unlist(strsplit(text, "(?<=[.!?])\\s+|(?<=다\\.)\\s+", perl = TRUE))
  sentences <- sentences[!grepl("벌금|추징|공탁|합의금|변제금|배상명령|소송비용|과태료",sentences)]
  money_pattern <- paste0("(금\\s*)?","(","[0-9.]+\\s*억(?:\\s*[0-9,]+\\s*만)?\\s*원",
                          "|","[0-9,]+\\s*천만\\s*원","|","[0-9,]+\\s*백만\\s*원","|",
                          "[0-9,]+\\s*만\\s*원","|","[0-9,]+\\s*원",")")
  
  #피해액 통일
  money_to_num <- function(x) {
    x <- gsub(",", "", x)
    x <- gsub("\\s+", "", x)
    x <- sub("^금", "", x)
    total <- 0
    
    #글자 -> 숫자로 바꾸기
    if (grepl("억", x)) {
      a <- sub("억.*", "", x)
      total <- total + as.numeric(a) * 100000000
      x <- sub("^[0-9.]+억", "", x)}
    
    if (grepl("천만", x)) {
      a <- sub("천만.*", "", x)
      total <- total + as.numeric(a) * 10000000
      x <- sub("^[0-9.]+천만", "", x)}
    
    if (grepl("백만", x)) {
      a <- sub("백만.*", "", x)
      total <- total + as.numeric(a) * 1000000
      x <- sub("^[0-9.]+백만", "", x)}
    
    if (grepl("만", x)) {
      a <- sub("만.*", "", x)
      total <- total + as.numeric(a) * 10000
      x <- sub("^[0-9.]+만", "", x)}
    
    if (total == 0) {
      a <- gsub("[^0-9.]", "", x)
      if (a != "") total <- as.numeric(a)}
    
    total
  }
  
  # 합계,총액이 명시
  total_sentences <- sentences[grepl("합계|총액|총 피해액|편취금액|피해금액",sentences) &grepl(money_pattern, sentences, perl = TRUE)]
  
  if (length(total_sentences) > 0) {
    amounts <- unlist(regmatches(total_sentences,gregexpr(money_pattern, total_sentences, perl = TRUE)))
    
    values <- sapply(amounts, money_to_num)
    values <- values[is.finite(values)]
    
    if (length(values) > 0) return(max(values))}
  
  # 피해액 문맥 ##판결문 확인 필요!!!
  damage_sentences <- sentences[
    grepl("편취|교부받|송금받|가로채|재산상 이익|피해자로부터|피해자들로부터",sentences) &grepl(money_pattern, sentences, perl = TRUE) ]
  
  if (length(damage_sentences) == 0) return(NA_real_)
  amounts <- unlist(regmatches(damage_sentences,gregexpr(money_pattern, damage_sentences, perl = TRUE)))
  
  values <- sapply(amounts, money_to_num)
  values <- values[is.finite(values) & values > 0]
  
  if (length(values) == 0) return(NA_real_)
  
  max(values)
}

demage <- sapply(eda_breach$full_text, extract_damage)
eda_breach$demage <- demage


#breach$repeated 관련문장 뽑은 후 직접 읽어서 태깅
keywords1 <- paste0(
  "(",paste(c("상습적\\w*","상습성","상습\\s*사기","상습범","누범",
      "동종\\s*(전과|범행|범죄|전력)",
      "사기\\s*(죄)?\\s*(전과|전력)",
      "사기죄로\\s*(처벌|벌금형|징역형|집행유예)[^.!?]{0,15}(받|선고)",
      "사기죄[^.!?]{0,15}(처벌|벌금형|징역형|집행유예)[^.!?]{0,15}(받|선고)"),
    collapse = "|"),")")
keywords2 <- paste0(
  "(",paste(c("초범","(동종\\s*)?(전과|전력)[은는이가을를\\s]*없",
              "(동종\\s*)?(전과|전력)[은는이가을를\\s]*전혀\\s*없",
              "아무런\\s*(전과|전력)[은는이가을를\\s]*없",
              "(형사처벌|처벌)[을를\\s]*받은\\s*전력[이가은는\\s]*없"),collapse = "|"),")")

for (i in 1:nrow(eda_breach)) {
  x <- eda_breach$full_text[i]
  
  if (is.na(x) || x == "") next
  
  x <- gsub("\n", " ", x)
  x <- gsub("\\s+", " ", x)
  
  sentences <- unlist(strsplit(x,"(?<=[.!?])\\s+|(?<=다\\.)\\s+",perl = TRUE))
  
  for (sentence in sentences) {
    
    positive_match <- grepl(keywords1,sentence,perl = TRUE)
    negative_match <- grepl(keywords2,sentence,perl = TRUE)
    
    if (positive_match && !negative_match) {
      cat(i,"\n",sentence,"\n\n")
    }
  }
}
repeated <- c(
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, NA,
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE,
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, NA, FALSE, FALSE, FALSE,
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
  FALSE, FALSE, NA, NA, NA, NA, FALSE, NA, FALSE, NA,
  FALSE, NA, FALSE, NA, FALSE, FALSE, FALSE, FALSE, FALSE, NA,
  FALSE, FALSE, NA, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE,
  FALSE, FALSE, FALSE, FALSE, NA, FALSE, NA, NA, FALSE, NA,
  NA, FALSE, FALSE, FALSE, FALSE, NA, FALSE, NA, NA, NA,
  NA, FALSE, NA, FALSE, NA, NA, FALSE, FALSE, NA, FALSE,
  NA, FALSE, NA, NA, FALSE, FALSE, NA, FALSE, NA, NA,
  FALSE, FALSE, FALSE, NA, FALSE, FALSE, FALSE, FALSE, NA, NA,
  FALSE, FALSE, FALSE, FALSE, FALSE, NA, NA, NA, FALSE, FALSE)
eda_breach$repeated <- repeated
eda_breach$repeated <- as.integer(replace(eda_breach$repeated, is.na(eda_breach$repeated), FALSE))


#breach$conceal 관련문장 뽑은 후 직접 읽어서 태깅
keywords1 <- paste0("(",paste(
    c("은닉\\w*","은폐\\w*","증거\\s*인멸\\w*","증거를\\s*인멸\\w*",
      "증거를?\\s*.{0,5}인멸\\w*","범행\\s*.{0,5}은폐\\w*","범행\\s*.{0,5}은닉\\w*",
      "기록\\s*.{0,5}삭제\\w*"),collapse = "|"),")")

for (i in 1:nrow(eda_breach)) {
  x <- eda_breach$full_text[i]
  
  if (is.na(x) || x == "") next
  
  x <- gsub("\n", " ", x)
  x <- gsub("\\s+", " ", x)
  
  sentences <- unlist(strsplit(x,"(?<=[.!?])\\s+|(?<=다\\.)\\s+",perl = TRUE))
  
  for (sentence in sentences) {
    
    positive_match <- grepl(keywords1,sentence,perl = TRUE)
    negative_match <- grepl(keywords2,sentence,perl = TRUE)
    
    if (positive_match && !negative_match) {
      cat(i,"\n",sentence,"\n\n")
    }
  }
}

conceal <- c(
  TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, TRUE,
  TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
  TRUE, NA, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, NA, NA,
  TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, NA, TRUE, TRUE,
  TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
  TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, NA, TRUE, TRUE, TRUE,
  TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
  NA, NA, TRUE, NA, NA, NA, NA, NA, TRUE, NA,
  TRUE, NA, TRUE, NA, NA, TRUE, NA, NA, TRUE, NA,
  TRUE, NA, NA, TRUE, TRUE, NA, TRUE, FALSE, NA, NA,
  TRUE, NA, NA, TRUE, NA, NA, NA, NA, FALSE, NA,
  NA, NA, NA, NA, NA, NA, NA, NA, TRUE, NA,
  NA, NA, NA, TRUE, NA, NA, NA, TRUE, NA, NA,
  NA, NA, NA, NA, TRUE, NA, NA, TRUE, NA, NA,
  NA, TRUE, TRUE, NA, TRUE, TRUE, TRUE, NA, NA, NA,
  NA, TRUE, NA, TRUE, NA, NA, NA, NA, NA, NA)
eda_breach$conceal <- 
eda_breach$conceal <- as.integer(replace(eda_breach$conceal, is.na(eda_breach$conceal), FALSE))

#breach$reflection #관련문장 뽑은 후 직접 읽어서 태깅


for (i in 1:nrow(eda_breach)) {
  x <- eda_breach$full_text[i]
  
  if (is.na(x) || x == "") next
  
  x <- gsub("\n", " ", x)
  x <- gsub("\\s+", " ", x)
  
  sentences <- unlist(strsplit(x,"(?<=[.!?])\\s+|(?<=다\\.)\\s+",perl = TRUE))
  
  for (sentence in sentences) {
    
    positive_match <- grepl(keywords1,sentence,perl = TRUE)
    negative_match <- grepl(keywords2,sentence,perl = TRUE)
    
    if (positive_match && !negative_match) {
      cat(i,"\n",sentence,"\n\n")
    }
  }
}

reflection <- c(
  TRUE, FALSE, TRUE, FALSE, NA, NA, NA, NA, TRUE, FALSE,
  FALSE, TRUE, FALSE, TRUE, TRUE, NA, NA, NA, NA, TRUE,
  NA, NA, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, NA, NA,
  FALSE, TRUE, FALSE, NA, NA, NA, NA, NA, TRUE, TRUE,
  FALSE, FALSE, FALSE, NA, NA, FALSE, NA, FALSE, FALSE, NA,
  NA, FALSE, FALSE, FALSE, NA, FALSE, NA, NA, FALSE, FALSE,
  FALSE, NA, NA, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, NA,
  FALSE, TRUE, FALSE, NA, NA, FALSE, TRUE, TRUE, FALSE, NA,
  FALSE, NA, FALSE, NA, NA, FALSE, NA, TRUE, NA, NA,
  FALSE, FALSE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, NA, TRUE,
  TRUE, FALSE, TRUE, NA, NA, FALSE, TRUE, FALSE, FALSE, NA,
  NA, NA, NA, FALSE, TRUE, FALSE, FALSE, NA, FALSE, NA,
  NA, FALSE, TRUE, NA, FALSE, NA, FALSE, NA, NA, TRUE,
  NA, TRUE, NA, NA, TRUE, FALSE, NA, FALSE, NA, NA,
  NA, FALSE, TRUE, NA, FALSE, FALSE, FALSE, TRUE, NA, NA,
  FALSE, FALSE, FALSE, FALSE, TRUE, NA, TRUE, NA, TRUE, TRUE)
eda_breach$reflection <- reflection
eda_breach$reflection <- as.integer(replace(eda_breach$reflection, is.na(eda_breach$reflection), FALSE))

#breach$settlement #관련문장 뽑은 후 직접 읽어서 태깅
keywords1 <- paste0("(",paste(
    c("피해자(들)?와\\s*원만히\\s*합의\\w*",
      "피해자(들)?와\\s*합의\\w*",
      "원만히\\s*합의\\w*",
      "합의금\\w*\\s*(지급|공탁|전달|교부|예치)\\w*",
      "합의에\\s*이르\\w*",
      "합의를\\s*하\\w*",
      "합의가\\s*성립\\w*",
      "합의서\\w*\\s*제출\\w*"),collapse = "|"),")")
keywords2 <- paste0("(",paste(c(
      "합의하지\\s*못\\w*",
      "합의되지\\s*않\\w*",
      "피해자(들)?와\\s*합의하지\\s*않\\w*",
      "합의에\\s*이르지\\s*못\\w*",
      "합의가\\s*이루어지지\\s*않\\w*",
      "합의\\s*불성립",
      "합의\\s*실패"),collapse = "|"),")")

for (i in 1:nrow(eda_breach)) {
  x <- eda_breach$full_text[i]
  
  if (is.na(x) || x == "") next
  
  x <- gsub("\n", " ", x)
  x <- gsub("\\s+", " ", x)
  
  sentences <- unlist(strsplit(x,"(?<=[.!?])\\s+|(?<=다\\.)\\s+",perl = TRUE))
  
  for (sentence in sentences) {
    
    positive_match <- grepl(keywords1,sentence,perl = TRUE)
    negative_match <- grepl(keywords2,sentence,perl = TRUE)
    
    if (positive_match && !negative_match) {
      cat(i,"\n",sentence,"\n\n")
    }
  }
}

settlement <- 
  c(TRUE, FALSE, NA, NA, NA, NA, NA, NA, NA, FALSE,
    FALSE, NA, NA, NA, NA, NA, NA, NA, NA, FALSE,
    NA, NA, NA, TRUE, FALSE, NA, NA, NA, NA, NA,
    NA, NA, NA, NA, NA, NA, NA, NA, NA, TRUE,
    FALSE, FALSE, NA, NA, NA, FALSE, NA, NA, NA, NA,
    NA, NA, FALSE, FALSE, NA, FALSE, NA, NA, FALSE, NA,
    NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,
    NA, NA, NA, NA, NA, NA, NA, FALSE, FALSE, NA,
    FALSE, NA, FALSE, NA, NA, FALSE, NA, FALSE, NA, NA,
    FALSE, NA, NA, TRUE, NA, FALSE, NA, FALSE, NA, NA,
    NA, FALSE, NA, TRUE, NA, NA, NA, NA, NA, NA,
    NA, NA, NA, NA, TRUE, NA, NA, NA, NA, NA,
    NA, NA, NA, NA, NA, NA, FALSE, NA, NA, FALSE,
    NA, NA, NA, NA, NA, FALSE, NA, NA, NA, NA,
    NA, NA, NA, NA, FALSE, NA, TRUE, TRUE, NA, NA,
    NA, TRUE, FALSE, NA, NA, NA, NA, NA, NA, NA)
eda_breach$settlement <- settlement
eda_breach$settlement <- as.integer(replace(eda_breach$settlement, is.na(eda_breach$settlement), FALSE))


#breach$restitution 관련문장 뽑은 후 직접 읽어서 태깅
keywords1 <- paste0("(",paste(
    c("변제\\w*",
      "피해액\\s*변제\\w*",
      "피해금\\s*변제\\w*",
      "편취금\\s*변제\\w*",
      "일부\\s*변제\\w*",
      "전부\\s*변제\\w*",
      "상당\\s*부분\\s*변제\\w*",
      "피해\\s*회복\\w*",
      "피해액\\s*회복\\w*",
      "피해가\\s*회복\\w*",
      "피해회복이\\s*이루어\\w*",
      "피해를\\s*회복\\w*",
      "피해액\\s*반환\\w*",
      "편취금\\s*반환\\w*",
      "피해자에게\\s*지급\\w*",
      "피해자에게\\s*반환\\w*",
      "배상\\w*",
      "변상\\w*",
      "상환\\w*",
      "환부\\w*"),collapse = "|"),")")

keywords2 <- paste0("(",paste(
    c("변제하지\\s*않\\w*",
      "변제하지\\s*아니\\w*",
      "변제하지\\s*못\\w*",
      "변제된\\s*바\\s*없\\w*",
      "피해\\s*회복되지\\s*않\\w*",
      "피해를\\s*회복하지\\s*못\\w*",
      "피해회복이\\s*이루어지지\\s*않\\w*",
      "반환하지\\s*않\\w*",
      "지급하지\\s*않\\w*",
      "배상하지\\s*않\\w*",
      "변상하지\\s*않\\w*",
      "상환하지\\s*않\\w*",
      "아무런\\s*(변제|배상|변상|상환|피해회복)"),
    collapse = "|"),")")

for (i in 1:nrow(eda_breach)) {
  x <- eda_breach$full_text[i]
  
  if (is.na(x) || x == "") next
  
  x <- gsub("\n", " ", x)
  x <- gsub("\\s+", " ", x)
  
  sentences <- unlist(strsplit(x,"(?<=[.!?])\\s+|(?<=다\\.)\\s+",perl = TRUE))
  
  for (sentence in sentences) {
    
    positive_match <- grepl(keywords1,sentence,perl = TRUE)
    negative_match <- grepl(keywords2,sentence,perl = TRUE)
    
    if (positive_match && !negative_match) {
      cat(i,"\n",sentence,"\n\n")
    }
  }
}

restitution <- c(
  TRUE, FALSE, NA, FALSE, NA, NA, NA, NA, TRUE, FALSE,
  TRUE, NA, NA, TRUE, TRUE, NA, NA, TRUE, TRUE, FALSE,
  NA, NA, TRUE, TRUE, TRUE, NA, NA, NA, NA, NA,
  NA, NA, TRUE, NA, TRUE, NA, NA, NA, TRUE, TRUE,
  FALSE, FALSE, NA, NA, NA, TRUE, NA, NA, FALSE, NA,
  NA, NA, FALSE, TRUE, NA, FALSE, NA, NA, FALSE, NA,
  NA, NA, NA, NA, NA, NA, NA, NA, TRUE, NA,
  TRUE, TRUE, NA, NA, NA, NA, TRUE, TRUE, TRUE, NA,
  NA, NA, TRUE, NA, TRUE, FALSE, NA, FALSE, NA, NA,
  FALSE, NA, TRUE, TRUE, TRUE, FALSE, NA, TRUE, NA, TRUE,
  TRUE, NA, NA, TRUE, NA, NA, NA, NA, NA, NA,
  NA, NA, NA, TRUE, TRUE, NA, NA, NA, FALSE, NA,
  NA, NA, NA, NA, NA, NA, TRUE, NA, NA, TRUE,
  NA, FALSE, NA, NA, TRUE, FALSE, NA, NA, NA, NA,
  NA, NA, TRUE, NA, FALSE, TRUE, TRUE, TRUE, NA, NA,
  TRUE, NA, TRUE, NA, NA, NA, FALSE, NA, TRUE, NA
)
eda_breach$restitution <- restitution
eda_breach$restitution<- as.integer(replace(eda_breach$restitution, is.na(eda_breach$restitution), FALSE))

#breach_감경여부

reduced_check <- function(x, y) {
  if (length(x) != length(y)) {
    stop("피해액 벡터 x와 형량 벡터 y의 길이가 같아야 합니다.")}
  
  result <- rep(NA_character_, length(x))
  
  result[
    !is.na(x) & !is.na(y) &
      ((x < 100000000 & y <= 4) |(x >= 100000000 & x < 500000000 & y <= 12) |
          (x >= 500000000 & x < 5000000000 & y <= 24) |
          (x >= 5000000000 & x < 30000000000 & y <= 48) |
          (x >= 30000000000 & y <= 60))] <- TRUE
  
  result[!is.na(x) & !is.na(y) & is.na(result)] <- FALSE
  
  return(result)}

eda_breach$outcome <- reduced_check(eda_breach$demage, eda_breach$month)

#use_breach
use_breach <- data.frame(
  outcome <- eda_breach$outcome,
  lawyer = eda_breach$lawyer,
  punishment = eda_breach$punishment,
  month = eda_breach$month,
  demage = eda_breach$demage,
  repeated = eda_breach$repeated,
  reflection = eda_breach$reflection,
  conceal = eda_breach$conceal,
  settlement = eda_breach$settlement,
  restitution = eda_breach$restitution )

use_breach$crime = "breach"
write.csv(use_breach, "Breach")




#eda_fraud
eda_fraud <- data[data$tag_crime_group_manual == "fraud",]
eda_fraud <- eda_fraud[eda_fraud$guilty == "Guilty",]
eda_fraud <- eda_fraud[eda_breach$punishment %in% c("both", "imprisonment", "suspended"),]

#fraud_month
sentence_to_year <- function(x) {
  year <- 0
  month <- 0
  
  if (grepl("[0-9]+년", x)) {
    year <- as.numeric(sub(".*?([0-9]+)년.*", "\\1", x))}
  
  if (grepl("[0-9]+개월", x)) {
    month <- as.numeric(sub(".*?([0-9]+)개월.*", "\\1", x))}
  
  else if (grepl("[0-9]+월", x)) {
    month <- as.numeric(sub(".*?([0-9]+)월.*", "\\1", x))}
  
  year*12 + month}

year <- c()
for (i in 1:nrow(eda_breach)) {
  x <- Guilty_FRAUD$full_text[i]
  x <- gsub("\n", "", x)
  x <- unlist(strsplit(x, split = " "))
  
  if (any(x == "징역", na.rm = TRUE)) {
    ind <- which(x == "징역")[1]
    year[i] <- sentence_to_year(paste0(x[ind:(ind+2)], collapse = ""))}
  
  else {year[i] <- NA}}

eda_fraud$month <- year

#fraud$demage
extract_damage <- function(text) {
  if (is.na(text) | text == "") return(NA)
  text <- gsub("\n", " ", text)
  sentences <- unlist(strsplit(text, "(?<=[.!?])\\s+|(?<=다\\.)\\s+", perl = TRUE))
  sentences <- sentences[!grepl("벌금|추징|공탁|합의금|변제금|배상명령|소송비용|과태료",sentences)]
  money_pattern <- paste0("(금\\s*)?","(","[0-9.]+\\s*억(?:\\s*[0-9,]+\\s*만)?\\s*원",
                          "|","[0-9,]+\\s*천만\\s*원","|","[0-9,]+\\s*백만\\s*원","|",
                          "[0-9,]+\\s*만\\s*원","|","[0-9,]+\\s*원",")")
  
  #피해액 통일
  money_to_num <- function(x) {
    x <- gsub(",", "", x)
    x <- gsub("\\s+", "", x)
    x <- sub("^금", "", x)
    total <- 0
    
    #글자 -> 숫자로 바꾸기
    if (grepl("억", x)) {
      a <- sub("억.*", "", x)
      total <- total + as.numeric(a) * 100000000
      x <- sub("^[0-9.]+억", "", x)}
    
    if (grepl("천만", x)) {
      a <- sub("천만.*", "", x)
      total <- total + as.numeric(a) * 10000000
      x <- sub("^[0-9.]+천만", "", x)}
    
    if (grepl("백만", x)) {
      a <- sub("백만.*", "", x)
      total <- total + as.numeric(a) * 1000000
      x <- sub("^[0-9.]+백만", "", x)}
    
    if (grepl("만", x)) {
      a <- sub("만.*", "", x)
      total <- total + as.numeric(a) * 10000
      x <- sub("^[0-9.]+만", "", x)}
    
    if (total == 0) {
      a <- gsub("[^0-9.]", "", x)
      if (a != "") total <- as.numeric(a)}
    
    total
  }
  
  # 합계,총액이 명시
  total_sentences <- sentences[grepl("합계|총액|총 피해액|편취금액|피해금액",sentences) &grepl(money_pattern, sentences, perl = TRUE)]
  
  if (length(total_sentences) > 0) {
    amounts <- unlist(regmatches(total_sentences,gregexpr(money_pattern, total_sentences, perl = TRUE)))
    
    values <- sapply(amounts, money_to_num)
    values <- values[is.finite(values)]
    
    if (length(values) > 0) return(max(values))}
  
  # 피해액 문맥 ##판결문 확인 필요!!!
  damage_sentences <- sentences[
    grepl("편취|교부받|송금받|가로채|재산상 이익|피해자로부터|피해자들로부터",sentences) &grepl(money_pattern, sentences, perl = TRUE) ]
  
  if (length(damage_sentences) == 0) return(NA_real_)
  amounts <- unlist(regmatches(damage_sentences,gregexpr(money_pattern, damage_sentences, perl = TRUE)))
  
  values <- sapply(amounts, money_to_num)
  values <- values[is.finite(values) & values > 0]
  
  if (length(values) == 0) return(NA_real_)
  
  max(values)
}

demage <- sapply(eda_fraud$full_text, extract_damage)
eda_fraud$demage <- demage


#fraud$repeated 관련문장 뽑은 후 직접 읽어서 태깅
keywords1 <- paste0(
  "(",paste(c("상습적\\w*","상습성","상습\\s*사기","상습범","누범",
              "동종\\s*(전과|범행|범죄|전력)",
              "사기\\s*(죄)?\\s*(전과|전력)",
              "사기죄로\\s*(처벌|벌금형|징역형|집행유예)[^.!?]{0,15}(받|선고)",
              "사기죄[^.!?]{0,15}(처벌|벌금형|징역형|집행유예)[^.!?]{0,15}(받|선고)"),
            collapse = "|"),")")
keywords2 <- paste0(
  "(",paste(c("초범","(동종\\s*)?(전과|전력)[은는이가을를\\s]*없",
              "(동종\\s*)?(전과|전력)[은는이가을를\\s]*전혀\\s*없",
              "아무런\\s*(전과|전력)[은는이가을를\\s]*없",
              "(형사처벌|처벌)[을를\\s]*받은\\s*전력[이가은는\\s]*없"),collapse = "|"),")")

for (i in 1:nrow(eda_breach)) {
  x <- eda_fraud$full_text[i]
  
  if (is.na(x) || x == "") next
  
  x <- gsub("\n", " ", x)
  x <- gsub("\\s+", " ", x)
  
  sentences <- unlist(strsplit(x,"(?<=[.!?])\\s+|(?<=다\\.)\\s+",perl = TRUE))
  
  for (sentence in sentences) {
    
    positive_match <- grepl(keywords1,sentence,perl = TRUE)
    negative_match <- grepl(keywords2,sentence,perl = TRUE)
    
    if (positive_match && !negative_match) {
      cat(i,"\n",sentence,"\n\n")
    }
  }
}
repeated <- c(
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,  # 1~10번
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,  # 11~20번
  FALSE, FALSE, FALSE, FALSE, FALSE, NA, FALSE, FALSE, TRUE, FALSE,  # 21~30번
  FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, NA, NA, FALSE,  # 31~40번
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,  # 41~50번
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,  # 51~60번
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,  # 61~70번
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,  # 71~80번
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,  # 81~90번
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE,  # 91~100번
  FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,  # 101~110번
  FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,  # 111~120번
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,  # 121~130번
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,  # 131~140번
  FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE,  # 141~150번
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,  # 151~160번
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,  # 161~170번
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,  # 171~180번
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,  # 181~190번
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,  # 191~200번
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,  # 201~210번
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE,  # 211~220번
  FALSE, NA, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,  # 221~230번
  FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,  # 231~240번
  FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE  # 241~248번
)

eda_fraud$repeated <- repeated
eda_fraud$repeated <- as.integer(replace(eda_fraud$repeated, is.na(eda_fraud$repeated), FALSE))


#fraud$conceal 관련문장 뽑은 후 직접 읽어서 태깅
keywords1 <- paste0("(",paste(
  c("은닉\\w*","은폐\\w*","증거\\s*인멸\\w*","증거를\\s*인멸\\w*",
    "증거를?\\s*.{0,5}인멸\\w*","범행\\s*.{0,5}은폐\\w*","범행\\s*.{0,5}은닉\\w*",
    "기록\\s*.{0,5}삭제\\w*"),collapse = "|"),")")

for (i in 1:nrow(eda_breach)) {
  x <- eda_breach$full_text[i]
  
  if (is.na(x) || x == "") next
  
  x <- gsub("\n", " ", x)
  x <- gsub("\\s+", " ", x)
  
  sentences <- unlist(strsplit(x,"(?<=[.!?])\\s+|(?<=다\\.)\\s+",perl = TRUE))
  
  for (sentence in sentences) {
    
    positive_match <- grepl(keywords1,sentence,perl = TRUE)
    negative_match <- grepl(keywords2,sentence,perl = TRUE)
    
    if (positive_match && !negative_match) {
      cat(i,"\n",sentence,"\n\n")
    }
  }
}

conceal <- c(
  TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, TRUE,  # 1~10번
  NA, NA, NA, NA, NA, NA, NA, NA, TRUE, NA,  # 11~20번
  NA, NA, NA, NA, TRUE, NA, FALSE, NA, NA, NA,  # 21~30번
  NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,  # 31~40번
  NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,  # 41~50번
  NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,  # 51~60번
  NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,  # 61~70번
  NA, NA, NA, NA, NA, TRUE, NA, NA, NA, NA,  # 71~80번
  TRUE, NA, NA, NA, NA, NA, NA, TRUE, NA, NA,  # 81~90번
  NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,  # 91~100번
  NA, NA, NA, NA, NA, NA, TRUE, NA, NA, NA,  # 101~110번
  NA, NA, NA, NA, TRUE, NA, TRUE, NA, NA, NA,  # 111~120번
  NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,  # 121~130번
  NA, NA, TRUE, NA, NA, NA, NA, NA, NA, NA,  # 131~140번
  NA, NA, NA, TRUE, NA, NA, NA, NA, NA, TRUE,  # 141~150번
  NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,  # 151~160번
  NA, NA, NA, NA, NA, NA, NA, NA, NA, TRUE,  # 161~170번
  NA, TRUE, TRUE, NA, NA, NA, NA, TRUE, NA, NA,  # 171~180번
  NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,  # 181~190번
  TRUE, NA, NA, NA, NA, NA, NA, NA, NA, NA,  # 191~200번
  NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,  # 201~210번
  NA, NA, NA, NA, NA, NA, NA, TRUE, NA, NA,  # 211~220번
  TRUE, NA, NA, TRUE, NA, NA, TRUE, NA, NA, NA,  # 221~230번
  NA, TRUE, NA, NA, NA, NA, NA, NA, NA, NA,  # 231~240번
  NA, NA, NA, NA, NA, NA, NA, NA  # 241~248번
)
eda_fraud$conceal <- conceal
eda_fraud$conceal <- as.integer(replace(eda_fraud$conceal, is.na(eda_fraud$conceal), FALSE))

#breach$reflection #관련문장 뽑은 후 직접 읽어서 태깅
keywords1 <- paste0("(",paste(c("반성\\w*","깊이\\s*반성\\w*","진지하게\\s*반성\\w*",
                                "잘못을\\s*뉘우치\\w*","범행을\\s*뉘우치\\w*","자신의\\s*잘못을\\s*인정\\w*",
                                "잘못을\\s*인정\\w*","범행을\\s*인정\\w*","범죄사실을\\s*인정\\w*",
                                "공소사실을\\s*인정\\w*","이\\s*사건\\s*범행을\\s*인정\\w*",
                                "자백\\w*","범행을\\s*자백\\w*","수사기관에서\\s*자백\\w*","법정에서\\s*자백\\w*",
                                "모든\\s*범행을\\s*시인\\w*","범행을\\s*시인\\w*","공소사실을\\s*시인\\w*",
                                "수사에\\s*협조\\w*","수사기관의\\s*조사에\\s*협조\\w*"),
                              collapse = "|"),")")

keywords2 <- paste0("(",paste(
    c("반성하지\\s*않\\w*",
      "반성하는\\s*태도를\\s*보이지\\s*않\\w*",
      "진지한\\s*반성의\\s*태도를\\s*보이지\\s*않\\w*",
      "반성의\\s*기미가\\s*없\\w*",
      "반성의\\s*태도가\\s*없\\w*",
      "잘못을\\s*뉘우치지\\s*않\\w*",
      "범행을\\s*부인\\w*",
      "범죄사실을\\s*부인\\w*",
      "공소사실을\\s*부인\\w*",
      "혐의를\\s*부인\\w*",
      "자백하지\\s*않\\w*",
      "범행을\\s*자백하지\\s*않\\w*",
      "변명으로\\s*일관\\w*",
      "책임을\\s*회피\\w*",
      "책임을\\s*전가\\w*",
      "납득하기\\s*어려운\\s*변명\\w*"),collapse = "|"),")")

for (i in 1:nrow(eda_breach)) {
  x <- eda_breach$full_text[i]
  
  if (is.na(x) || x == "") next
  
  x <- gsub("\n", " ", x)
  x <- gsub("\\s+", " ", x)
  
  sentences <- unlist(strsplit(x,"(?<=[.!?])\\s+|(?<=다\\.)\\s+",perl = TRUE))
  
  for (sentence in sentences) {
    
    positive_match <- grepl(keywords1,sentence,perl = TRUE)
    negative_match <- grepl(keywords2,sentence,perl = TRUE)
    
    if (positive_match && !negative_match) {
      cat(i,"\n",sentence,"\n\n")
    }
  }
}

reflection <- c(
  TRUE, FALSE, TRUE, FALSE, NA, NA, NA, NA, TRUE, FALSE,  # 1~10번
  NA, TRUE, TRUE, FALSE, TRUE, NA, NA, NA, TRUE, FALSE,  # 11~20번
  TRUE, NA, NA, NA, FALSE, NA, FALSE, NA, NA, NA,  # 21~30번
  NA, NA, NA, NA, NA, FALSE, FALSE, NA, NA, NA,  # 31~40번
  NA, NA, TRUE, NA, NA, NA, NA, NA, FALSE, NA,  # 41~50번
  NA, FALSE, NA, NA, NA, NA, FALSE, FALSE, TRUE, NA,  # 51~60번
  TRUE, TRUE, TRUE, NA, NA, NA, NA, FALSE, NA, NA,  # 61~70번
  NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,  # 71~80번
  NA, NA, NA, TRUE, FALSE, NA, TRUE, NA, TRUE, NA,  # 81~90번
  FALSE, NA, TRUE, NA, NA, NA, FALSE, NA, NA, NA,  # 91~100번
  NA, NA, NA, NA, NA, NA, NA, NA, FALSE, NA,  # 101~110번
  NA, NA, NA, NA, FALSE, NA, NA, NA, NA, NA,  # 111~120번
  NA, NA, NA, TRUE, NA, TRUE, NA, NA, NA, TRUE,  # 121~130번
  NA, NA, NA, NA, FALSE, FALSE, TRUE, NA, TRUE, NA,  # 131~140번
  FALSE, FALSE, NA, NA, TRUE, TRUE, NA, NA, NA, FALSE,  # 141~150번
  FALSE, FALSE, FALSE, NA, FALSE, NA, NA, NA, TRUE, NA,  # 151~160번
  NA, NA, FALSE, NA, NA, TRUE, NA, NA, NA, TRUE,  # 161~170번
  NA, NA, TRUE, NA, NA, NA, NA, FALSE, NA, NA,  # 171~180번
  NA, NA, NA, TRUE, NA, NA, TRUE, TRUE, TRUE, NA,  # 181~190번
  FALSE, NA, NA, NA, FALSE, NA, NA, NA, NA, NA,  # 191~200번
  FALSE, NA, NA, NA, TRUE, NA, NA, TRUE, NA, FALSE,  # 201~210번
  NA, NA, FALSE, FALSE, FALSE, NA, TRUE, TRUE, TRUE, FALSE,  # 211~220번
  TRUE, NA, NA, NA, FALSE, NA, TRUE, NA, NA, FALSE,  # 221~230번
  NA, FALSE, NA, NA, NA, NA, NA, NA, NA, TRUE,  # 231~240번
  NA, NA, NA, NA, FALSE, TRUE, FALSE, NA  # 241~248번
)
eda_fraud$reflection <- reflection
eda_fraud$reflection <- as.integer(replace(eda_fraud$reflection, is.na(eda_fraud$reflection), FALSE))

#breach$settlement #관련문장 뽑은 후 직접 읽어서 태깅
keywords1 <- paste0("(",paste(
  c("피해자(들)?와\\s*원만히\\s*합의\\w*",
    "피해자(들)?와\\s*합의\\w*",
    "원만히\\s*합의\\w*",
    "합의금\\w*\\s*(지급|공탁|전달|교부|예치)\\w*",
    "합의에\\s*이르\\w*",
    "합의를\\s*하\\w*",
    "합의가\\s*성립\\w*",
    "합의서\\w*\\s*제출\\w*"),collapse = "|"),")")
keywords2 <- paste0("(",paste(c(
  "합의하지\\s*못\\w*",
  "합의되지\\s*않\\w*",
  "피해자(들)?와\\s*합의하지\\s*않\\w*",
  "합의에\\s*이르지\\s*못\\w*",
  "합의가\\s*이루어지지\\s*않\\w*",
  "합의\\s*불성립",
  "합의\\s*실패"),collapse = "|"),")")

for (i in 1:nrow(eda_breach)) {
  x <- eda_breach$full_text[i]
  
  if (is.na(x) || x == "") next
  
  x <- gsub("\n", " ", x)
  x <- gsub("\\s+", " ", x)
  
  sentences <- unlist(strsplit(x,"(?<=[.!?])\\s+|(?<=다\\.)\\s+",perl = TRUE))
  
  for (sentence in sentences) {
    
    positive_match <- grepl(keywords1,sentence,perl = TRUE)
    negative_match <- grepl(keywords2,sentence,perl = TRUE)
    
    if (positive_match && !negative_match) {
      cat(i,"\n",sentence,"\n\n")
    }
  }
}

settlement <- c(
  TRUE, FALSE, NA, NA, NA, NA, NA, NA, NA, FALSE,  # 1~10번
  NA, TRUE, TRUE, FALSE, TRUE, NA, TRUE, NA, NA, FALSE,  # 11~20번
  NA, NA, NA, NA, TRUE, NA, NA, NA, NA, TRUE,  # 21~30번
  NA, NA, TRUE, NA, NA, NA, FALSE, NA, NA, NA,  # 31~40번
  NA, NA, NA, NA, NA, NA, NA, FALSE, NA, NA,  # 41~50번
  NA, NA, NA, NA, NA, NA, NA, NA, TRUE, NA,  # 51~60번
  FALSE, TRUE, NA, TRUE, NA, TRUE, NA, NA, NA, NA,  # 61~70번
  NA, NA, NA, NA, TRUE, TRUE, NA, NA, NA, NA,  # 71~80번
  NA, NA, NA, NA, NA, NA, TRUE, NA, NA, NA,  # 81~90번
  TRUE, FALSE, NA, NA, NA, NA, NA, NA, NA, TRUE,  # 91~100번
  FALSE, NA, TRUE, NA, NA, FALSE, NA, NA, NA, NA,  # 101~110번
  TRUE, NA, NA, NA, NA, NA, NA, NA, NA, NA,  # 111~120번
  TRUE, NA, NA, NA, NA, NA, NA, NA, TRUE, NA,  # 121~130번
  NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,  # 131~140번
  FALSE, TRUE, FALSE, TRUE, TRUE, NA, FALSE, NA, NA, TRUE,  # 141~150번
  NA, NA, NA, NA, NA, NA, FALSE, NA, NA, NA,  # 151~160번
  NA, TRUE, NA, FALSE, NA, NA, FALSE, NA, NA, NA,  # 161~170번
  NA, NA, NA, NA, TRUE, NA, NA, FALSE, FALSE, NA,  # 171~180번
  NA, NA, FALSE, TRUE, NA, NA, NA, NA, TRUE, NA,  # 181~190번
  NA, NA, NA, NA, NA, NA, FALSE, NA, NA, NA,  # 191~200번
  TRUE, NA, NA, NA, TRUE, NA, NA, TRUE, NA, FALSE,  # 201~210번
  NA, NA, NA, TRUE, NA, NA, NA, NA, TRUE, NA,  # 211~220번
  NA, NA, NA, NA, NA, NA, TRUE, NA, NA, TRUE,  # 221~230번
  NA, TRUE, TRUE, NA, NA, NA, NA, NA, NA, NA,  # 231~240번
  NA, NA, NA, NA, NA, TRUE, FALSE, TRUE  # 241~248번
)
eda_fraud$settlement <- settlement
eda_fraud$settlement <- as.integer(replace(eda_fraud$settlement, is.na(eda_fraud$settlement), FALSE))


#breach$restitution 관련문장 뽑은 후 직접 읽어서 태깅
keywords1 <- paste0("(",paste(
  c("변제\\w*",
    "피해액\\s*변제\\w*",
    "피해금\\s*변제\\w*",
    "편취금\\s*변제\\w*",
    "일부\\s*변제\\w*",
    "전부\\s*변제\\w*",
    "상당\\s*부분\\s*변제\\w*",
    "피해\\s*회복\\w*",
    "피해액\\s*회복\\w*",
    "피해가\\s*회복\\w*",
    "피해회복이\\s*이루어\\w*",
    "피해를\\s*회복\\w*",
    "피해액\\s*반환\\w*",
    "편취금\\s*반환\\w*",
    "피해자에게\\s*지급\\w*",
    "피해자에게\\s*반환\\w*",
    "배상\\w*",
    "변상\\w*",
    "상환\\w*",
    "환부\\w*"),collapse = "|"),")")

keywords2 <- paste0("(",paste(
  c("변제하지\\s*않\\w*",
    "변제하지\\s*아니\\w*",
    "변제하지\\s*못\\w*",
    "변제된\\s*바\\s*없\\w*",
    "피해\\s*회복되지\\s*않\\w*",
    "피해를\\s*회복하지\\s*못\\w*",
    "피해회복이\\s*이루어지지\\s*않\\w*",
    "반환하지\\s*않\\w*",
    "지급하지\\s*않\\w*",
    "배상하지\\s*않\\w*",
    "변상하지\\s*않\\w*",
    "상환하지\\s*않\\w*",
    "아무런\\s*(변제|배상|변상|상환|피해회복)"),
  collapse = "|"),")")

for (i in 1:nrow(eda_breach)) {
  x <- eda_breach$full_text[i]
  
  if (is.na(x) || x == "") next
  
  x <- gsub("\n", " ", x)
  x <- gsub("\\s+", " ", x)
  
  sentences <- unlist(strsplit(x,"(?<=[.!?])\\s+|(?<=다\\.)\\s+",perl = TRUE))
  
  for (sentence in sentences) {
    
    positive_match <- grepl(keywords1,sentence,perl = TRUE)
    negative_match <- grepl(keywords2,sentence,perl = TRUE)
    
    if (positive_match && !negative_match) {
      cat(i,"\n",sentence,"\n\n")
    }
  }
}

restitution <- c(
  TRUE, FALSE, NA, FALSE, NA, NA, NA, NA, TRUE, FALSE,  # 1~10번
  TRUE, NA, NA, NA, FALSE, NA, TRUE, NA, FALSE, TRUE,  # 11~20번
  NA, NA, NA, NA, TRUE, NA, NA, NA, NA, NA,  # 21~30번
  NA, NA, NA, NA, NA, NA, FALSE, NA, NA, NA,  # 31~40번
  NA, NA, FALSE, NA, NA, NA, NA, NA, NA, NA,  # 41~50번
  NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,  # 51~60번
  FALSE, NA, TRUE, TRUE, NA, NA, NA, FALSE, NA, FALSE,  # 61~70번
  NA, NA, NA, NA, TRUE, NA, TRUE, FALSE, NA, NA,  # 71~80번
  NA, TRUE, NA, FALSE, FALSE, NA, TRUE, FALSE, NA, NA,  # 81~90번
  NA, FALSE, NA, NA, NA, NA, FALSE, NA, NA, NA,  # 91~100번
  NA, NA, NA, NA, NA, NA, TRUE, NA, FALSE, NA,  # 101~110번
  TRUE, NA, NA, NA, NA, TRUE, NA, TRUE, NA, NA,  # 111~120번
  NA, NA, NA, NA, NA, FALSE, FALSE, FALSE, NA, FALSE,  # 121~130번
  NA, NA, NA, FALSE, FALSE, NA, NA, NA, NA, NA,  # 131~140번
  FALSE, NA, FALSE, NA, NA, NA, FALSE, NA, TRUE, TRUE,  # 141~150번
  NA, NA, NA, NA, NA, NA, NA, NA, TRUE, NA,  # 151~160번
  NA, NA, NA, TRUE, NA, NA, FALSE, NA, NA, TRUE,  # 161~170번
  NA, NA, TRUE, NA, NA, NA, NA, FALSE, NA, NA,  # 171~180번
  NA, NA, NA, NA, NA, NA, NA, FALSE, NA, NA,  # 181~190번
  FALSE, NA, NA, NA, NA, NA, FALSE, NA, NA, NA,  # 191~200번
  FALSE, NA, NA, FALSE, NA, NA, NA, TRUE, NA, FALSE,  # 201~210번
  NA, NA, NA, FALSE, NA, NA, NA, NA, NA, NA,  # 211~220번
  TRUE, NA, NA, NA, NA, NA, NA, NA, NA, TRUE,  # 221~230번
  NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,  # 231~240번
  NA, NA, NA, NA, NA, TRUE, NA, NA  # 241~248번
)
eda_fraud$restitution <- restitution
eda_fraud$restitution<- as.integer(replace(eda_fraud$restitution, is.na(eda_fraud$restitution), FALSE))

#eda_감경여부
#(1) data에서 글자만 모은 vector-list 만들기
lst <- list()
x<-c()
for (i in nrow(eda_fraud)) {
  x <- eda_fraud$full_text[i]
  gsub("\n", "", x)
  x <- unlist(strsplit(x, split = " "))
  lst[[i]] <- x  ##lst - FRAUD(사기만 모은 df)의 full_text의 글자벡터 리스트
}

#(2) 일차적으로 조직범죄 여부 판단하기
keywords <- c("총책","조직원","범죄조직","범죄단체","관리책","모집책","유인책",
              "현금수거책","수거책","인출책","전달책","송금책","역할분담","해외조직")

x<-c()
for (i in 1:nrow(eda_fraud)) {
  if (any(lst[[i]] %in% keywords)) {x<-c(x,TRUE)}
  else {x <- c(x,FALSE)}
}

eda_fraud$organization <- x

check_reduction_months <- function(x, y, z) {
  e1 <- 100000000
  e5 <- 500000000
  e50 <- 50000000000
  e300 <- 300000000000
  
  n <- length(x)
  max_reduction_months <- numeric(n)
  is_reduced <- logical(n)
  description <- character(n)
  
  for (i in seq_len(n)) {
    dmg <- x[i]
    sentence_months <- y[i]
    is_org <- z[i]
    
    if (is.na(dmg) || is.na(sentence_months) || is.na(is_org)) {
      max_reduction_months[i] <- NA
      description[i] <- NA
      is_reduced[i] <- NA
      next
    }
    
    if (is_org) {
      if (dmg < e1) {
        limit_months <- 18; desc_text <- "18개월 미만"
      } else if (dmg < e5) {
        limit_months <- 24; desc_text <- "2년 미만 (24개월)"
      } else if (dmg < e50) {
        limit_months <- 48; desc_text <- "4년 미만 (48개월)"
      } else if (dmg < e300) {
        limit_months <- 72; desc_text <- "6년 미만 (72개월)"
      } else {
        limit_months <- 96; desc_text <- "8년 미만 (96개월)"
      }
    } else {
      if (dmg < e1) {
        limit_months <- 6; desc_text <- "6개월 미만"
      } else if (dmg < e5) {
        limit_months <- 12; desc_text <- "1년 미만 (12개월)"
      } else if (dmg < e50) {
        limit_months <- 36; desc_text <- "3년 미만 (36개월)"
      } else if (dmg < e300) {
        limit_months <- 60; desc_text <- "5년 미만 (60개월)"
      } else {
        limit_months <- 72; desc_text <- "6년 미만 (72개월)"
      }
    }
    
    max_reduction_months[i] <- limit_months
    description[i] <- desc_text
    
    # 2. 감경 대상 여부 판단 (미만 조건)
    if (sentence_months < limit_months) {
      is_reduced[i] <- TRUE
    } else {
      is_reduced[i] <- FALSE
    }
  }
  
  return(data.frame(
    x_damage = x, 
    y_sentence_months = y, 
    z_org = z, 
    limit_criteria = description,
    is_reduced = is_reduced
  ))
}

result <- check_reduction_months(eda_fraud$demage, eda_fraud$month, eda_fraud$organization)
eda_fraud$outcome <- result$is_reduced

use_fraud <- data.frame(
  outcome <- eda_fraud$outcome,
  lawyer = eda_fraud$lawyer,
  punishment = eda_fraud$punishment,
  month = eda_fraud$month,
  demage = eda_fraud$demage,
  repeated = eda_fraud$repeated,
  reflection = eda_fraud$reflection,
  conceal = eda_fraud$conceal,
  settlement = eda_fraud$settlement,
  restitution = eda_fraud$restitution )

use_fraud$crime = "fraud"
write.csv(use_fraud, "Fraud")

#combined data
use_data <- rbind(use_fraud, use_theft, use_breach)
write.csv(use_data,"Usedata")

names(use_theft)
names(use_fraud)
names(use_breach)

names(use_fraud)[1] = "outcome"



use_breach <- data.frame(
  lawyer = eda_breach$lawyer,
  punishment = eda_breach$punishment,
  month = eda_breach$month,
  demage = eda_breach$demage,
  repeated = eda_breach$repeated,
  reflection = eda_breach$reflection,
  conceal = eda_breach$conceal,
  settlement = eda_breach$settlement,
  restitution = eda_breach$restitution )

use_theft$crime <- "theft"; use_breach$crime <- "breach"
use_theft_breach <- rbind(use_theft,use_breach)
use_theft_breach

write.csv(use_theft, "Theft")
