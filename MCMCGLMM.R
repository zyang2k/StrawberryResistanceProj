# --- 1. Data Preparation ------------------------

library(readr)
library(dplyr)
library(lubridate)
library(MCMCglmm)

# Load Data
X2024_Consolidated_BrixAcid <- read_csv("~/Desktop/Winter 2025/STA260/StrawberryResistance/2024_Consolidated_BrixAcid.csv")
Clean_Bot_Ratings          <- read_csv("~/Desktop/Winter 2025/STA260/StrawberryResistance/Clean_Bot_Ratings.csv")
Clean_Col_Ratings          <- read_csv("~/Desktop/Winter 2025/STA260/StrawberryResistance/Clean_Col_Ratings.csv")
Clean_NPT_Ratings          <- read_csv("~/Desktop/Winter 2025/STA260/StrawberryResistance/Clean_NPT_Ratings.csv")
Firmness                   <- read_csv("~/Desktop/Winter 2025/STA260/StrawberryResistance/Firmness.csv")

# Clean Bot, Col, and NPT datasets:
bot_clean <- Clean_Bot_Ratings %>%
  rename(Harvest = HarvestPeriod) %>%
  mutate(HarvestDate = as.Date(HarvestDate, format = "%Y-%m-%d")) %>%
  select(Entry, Block, HarvestDate, Harvest, TrayID, Position, Score, Disease)

col_clean <- Clean_Col_Ratings %>%
  rename(Harvest = HarvestPeriod) %>%
  mutate(HarvestDate = as.Date(HarvestDate, format = "%Y-%m-%d")) %>%
  select(Entry, Block, HarvestDate, Harvest, TrayID, Position, Score, Disease)

npt_clean <- Clean_NPT_Ratings %>%
  rename(Harvest = HarvestPeriod) %>%
  mutate(HarvestDate = as.Date(HarvestDate, format = "%Y-%m-%d")) %>%
  select(Entry, Block, HarvestDate, Harvest, TrayID, Position, Score, Disease)

# Clean Brix/Acid dataset (rename CorrectedTitro to Titro)
brix_acid_clean <- X2024_Consolidated_BrixAcid %>%
  rename(HarvestDate = Harvest_Date, Titro = CorrectedTitro) %>%
  mutate(HarvestDate = as.Date(HarvestDate, format = "%Y-%m-%d")) %>%
  select(Entry, Block, HarvestDate, Plot, Brix, Titro)

# Clean Firmness dataset (rename Date to HarvestDate, HarvestPeriod to Harvest, Firmness.kg to Firmness)
firmness_clean <- Firmness %>%
  rename(HarvestDate = Date, Harvest = HarvestPeriod, Firmness = Firmness.kg) %>%
  mutate(HarvestDate = as.Date(HarvestDate, format = "%Y-%m-%d")) %>%
  select(Entry, Block, HarvestDate, Plot, Firmness, Harvest)

# Aggregate quality measures
firmness_agg <- firmness_clean %>%
  group_by(Entry, Block, HarvestDate, Harvest, Plot) %>%
  summarize(Firmness = mean(Firmness, na.rm = TRUE), .groups = "drop")

brix_acid_agg <- brix_acid_clean %>%
  group_by(Entry, Block, HarvestDate) %>%
  summarize(
    Brix = mean(Brix, na.rm = TRUE),
    Titro = mean(Titro, na.rm = TRUE),
    .groups = "drop"
  )

# Combine disease datasets
disease_combined <- bind_rows(bot_clean, col_clean, npt_clean)

# Merge quality data
merged_data <- disease_combined %>%
  left_join(brix_acid_agg, by = c("Entry", "Block", "HarvestDate")) %>%
  left_join(firmness_agg, by = c("Entry", "Block", "HarvestDate", "Harvest")) %>%
  select(Entry, Block, HarvestDate, Harvest, Score, Disease, Brix, Titro, Firmness)

# Convert variables to proper types
merged_data <- merged_data %>%
  mutate(
    Score = ordered(Score),
    Block = as.factor(Block),
    Harvest = as.factor(Harvest),
    Disease = as.factor(Disease),
    Entry = as.factor(Entry)
  )

# Filter complete cases for the key predictors
merged_data_df <- merged_data %>%
  filter(complete.cases(Entry, Disease, Brix, Titro, Firmness)) %>%
  as.data.frame()

# Ensure only cultivars with observations in all 3 blocks and both harvests are retained
complete_cultivars <- merged_data_df %>%
  group_by(Entry) %>%
  summarize(nBlocks = n_distinct(Block), nHarvest = n_distinct(Harvest)) %>%
  filter(nBlocks == 3, nHarvest == 2) %>%
  pull(Entry)

merged_data_df <- merged_data_df %>%
  filter(Entry %in% complete_cultivars)

# --- 2. Split Data by Disease Type -------------------------------

data_bot <- merged_data_df %>% filter(Disease == "B")
data_col <- merged_data_df %>% filter(Disease == "C")
data_npt <- merged_data_df %>% filter(Disease == "N")

# --- 3. Define Prior (same for all models) ------------------------

prior_fixed <- list(
  R = list(V = 1, fix = 1),
  G = list(
    G1 = list(V = 1, nu = 0.002),   # Random effect: Block
    G2 = list(V = 1, nu = 0.002)    # Random effect: Harvest
  )
)

# --- 4. Fit Separate MCMC GLMMs for Each Disease ------------------

# Note: Since data are stratified by Disease, we remove Disease from the fixed effects.
formula_strat <- Score ~ Entry + Brix + Titro + Firmness

mcmc_model_bot <- MCMCglmm(
  formula_strat,
  random = ~ Block + Harvest,
  family = "ordinal",
  data = data_bot,
  prior = prior_fixed,
  singular.ok = TRUE,
  nitt = 13000, burnin = 3000, thin = 10
)

mcmc_model_col <- MCMCglmm(
  formula_strat,
  random = ~ Block + Harvest,
  family = "ordinal",
  data = data_col,
  prior = prior_fixed,
  singular.ok = TRUE,
  nitt = 13000, burnin = 3000, thin = 10
)

mcmc_model_npt <- MCMCglmm(
  formula_strat,
  random = ~ Block + Harvest,
  family = "ordinal",
  data = data_npt,
  prior = prior_fixed,
  singular.ok = TRUE,
  nitt = 13000, burnin = 3000, thin = 10
)

# --- 5. Summaries of the Three Models -----------------------------

summary(mcmc_model_bot)
summary(mcmc_model_col)
summary(mcmc_model_npt)



# --- 6. Naive LMM model -----

# Load required libraries
library(lme4)
library(performance)
library(gt)

# Fit LMM for Botrytis (BOT) disease
lmm_bot <- lmer(Score ~ Brix + Titro + Firmness + (1 | Entry) + (1 | Block) + (1 | Harvest), 
                data = data_bot)

# Model summary
summary(lmm_bot)

