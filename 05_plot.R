library(ggplot2)
library(dplyr)

d <- read.csv("weekly_nsi_analysis.csv", check.names = FALSE)

time_col <- names(d)[1]
d[[time_col]] <- as.Date(d[[time_col]])

z_score <- function(x) {
  (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
}

d <- d %>%
  arrange(.data[[time_col]]) %>%
  mutate(
    alarms_z = z_score(alarms),
    NSI_z = z_score(NSI),
    anxiety_z = z_score(anxiety)
  )

plot_data <- data.frame(
  date = rep(d[[time_col]], 3),
  series = rep(
    c(
      "Alarms - physical threat",
      "Media coverage - NSI",
      "Anxiety searches - חרדה"
    ),
    each = nrow(d)
  ),
  value = c(
    d$alarms_z,
    d$NSI_z,
    d$anxiety_z
  )
)

plot_data$date <- as.Date(plot_data$date)

my_colors <- c(
  "Alarms - physical threat" = "#c31e23",
  "Media coverage - NSI" = "#0d7d87",
  "Anxiety searches - חרדה" = "white"
)

x_breaks <- seq(
  from = min(plot_data$date, na.rm = TRUE),
  to = max(plot_data$date, na.rm = TRUE),
  length.out = 16
)

p <- ggplot(plot_data, aes(x = date, y = value, color = series)) +
  geom_line(linewidth = 1.25, alpha = 0.95) +
  geom_hline(
    yintercept = 0,
    color = "#777777",
    linewidth = 0.7,
    linetype = "dashed"
  ) +
  scale_color_manual(values = my_colors) +
  scale_x_date(
    breaks = x_breaks,
    labels = format(x_breaks, "%m/%Y")
  ) +
  labs(
    title = "Three Layers of Wartime Stress in Israel, 2023–2025",
    subtitle = "Weekly standardized indicators: physical threat, media coverage, and anxiety searches",
    x = "Month",
    y = "z-score of weekly alarms, NSI, and anxiety searches",
    color = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.background = element_rect(fill = "#1f2428", color = NA),
    panel.background = element_rect(fill = "#1f2428", color = NA),
    panel.grid.major = element_line(color = "#3a3f44", linewidth = 0.35),
    panel.grid.minor = element_blank(),
    plot.title = element_text(
      hjust = 0.5,
      face = "bold",
      size = 17,
      color = "white"
    ),
    plot.subtitle = element_text(
      hjust = 0.5,
      size = 11,
      color = "#cfcfcf"
    ),
    axis.title.x = element_text(color = "#d9d9d9", size = 12, margin = margin(t = 10)),
    axis.title.y = element_text(color = "#d9d9d9", size = 12, margin = margin(r = 10)),
    axis.text.x = element_text(color = "#d9d9d9", angle = 45, hjust = 1, size = 9),
    axis.text.y = element_text(color = "#d9d9d9", size = 10),
    legend.position = "top",
    legend.background = element_rect(fill = "#1f2428", color = NA),
    legend.key = element_rect(fill = "#1f2428", color = NA),
    legend.text = element_text(color = "#eeeeee", size = 10),
    panel.border = element_blank()
  )

print(p)

ggsave(
  filename = "three_layer_dark_style.png",
  plot = p,
  width = 13,
  height = 5.5,
  dpi = 150,
  bg = "#1f2428"
)