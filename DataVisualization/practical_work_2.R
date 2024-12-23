# Load required libraries
library(sf)
library(leaflet)
library(shiny)
library(dplyr)

# File paths to the CSV files
boundaries_csv <- "data/chicago_boundaries.csv"
chicago_csv <- "data/crimes_cleaned_any.csv"
police_district_csv <- "data/police_district_boundaries_cleaned.csv"

# Load the data
chicago_data_boundaries <- read.csv(boundaries_csv)
chicago_data <- read.csv(chicago_csv, check.names = FALSE) # Ensure column names are not altered
police_district_data <- read.csv(police_district_csv, check.names = FALSE) 

# Ensure 'Districts' and 'DIST_NUM' are the same type for the join
chicago_data$Districts <- as.integer(chicago_data$District)
police_district_data$DIST_NUM <- as.integer(police_district_data$DIST_NUM)

# Convert police district boundaries to an sf object
police_district_boundaries <- st_as_sf(police_district_data, wkt = "the_geom", crs = 4326)

# Extract months, days, and years from the data
chicago_data$Month <- format(as.Date(chicago_data$Date, format = "%m/%d/%Y"), "%B") # Add a Month column
chicago_data$Month <- factor(chicago_data$Month, levels = month.name) # Order months correctly
chicago_data$Day <- format(as.Date(chicago_data$Date, format = "%m/%d/%Y"), "%d")   # Add a Day column
chicago_data$Day <- sprintf("%02d", as.numeric(chicago_data$Day)) # Ensure days are two-digit and ordered
chicago_data$Year <- format(as.Date(chicago_data$Date, format = "%m/%d/%Y"), "%Y")  # Extract Year

# Extract year-specific crime data into separate columns for clarity
police_district_boundaries$Count_2022 <- police_district_boundaries$crimes_2022
police_district_boundaries$Count_2023 <- police_district_boundaries$crimes_2023
police_district_boundaries$Total_Count <- police_district_boundaries$total_crimes

# Define the UI
ui <- fluidPage(
  # Title of the application
  titlePanel("Practical Work Assignment 2 - Chicago Crime"),
  
  # Tabset panel to separate sections
  tabsetPanel(
    # Tab for Question 1
    tabPanel("Question 1", "Content for Question 1"),
    
    # Tab for Question 2
    tabPanel("Question 2"),
    
    # Tab for Chicago Map
    tabPanel(
      "Chicago Map",
      fluidRow(
        # Left column for filters
        column(
          width = 3,
          h4("Map Filters"),
          sliderInput("opacity", "Fill Opacity:", min = 0, max = 1, value = 0.3, step = 0.1),
          selectInput(
            "crime_type", "Select Crime Type:",
            choices = c("All", unique(chicago_data$`Primary Type`)),  # Add "All" for all crime types
            selected = "PUBLIC INDECENCY"  # Default crime type
          ),
          selectInput(
            "month", "Select Month:",
            choices = c("All", levels(chicago_data$Month)),
            selected = "All"
          ),
          selectInput(
            "day", "Select Day:",
            choices = c("All", sort(unique(chicago_data$Day))),
            selected = "All"
          ),
          selectInput(
            "year", "Select Year:",
            choices = c("All", unique(chicago_data$Year)),
            selected = "All"
          ),
          selectInput(
            "district", "Select District:",
            choices = c("All", sort(unique(chicago_data$Districts))),
            selected = "All"
          )
        ),
        
        # Right column for the map
        column(
          width = 9,
          leafletOutput("district_map", height = "700px")
        )
      )
    ),
    
    # Tab for Heatmap
    tabPanel(
      "Heatmap",
      fluidRow(
        column(
          width = 3,
          h4("Heatmap Filters"),
          selectInput(
            "year_heatmap", "Select Year:",
            choices = c("2022", "2023", "All"),
            selected = "All"
          ),
          sliderInput(
            "crime_range", "Select Crime Range:",
            min = 0, max = max(police_district_boundaries$Total_Count, na.rm = TRUE),
            value = c(0, max(police_district_boundaries$Total_Count, na.rm = TRUE)),
            step = 1000
          )
        ),
        column(
          width = 9,
          leafletOutput("heatmap", height = "700px")
        )
      )
    )
  )
)

# Define the server logic
server <- function(input, output, session) {
  # Render the police district map with crime points
  output$district_map <- renderLeaflet({
    # Filter crime data based on user inputs
    filtered_data <- chicago_data %>%
      filter(
        (input$crime_type == "All" | `Primary Type` == input$crime_type),
        (input$month == "All" | Month == input$month),
        (input$day == "All" | Day == input$day),
        (input$year == "All" | Year == input$year),
        (input$district == "All" | Districts == input$district)
      )
    
    # Render the map with police district boundaries and points
    leaflet(data = police_district_boundaries) %>%
      addTiles() %>%
      addPolygons(
        fillColor = "blue",  # Fixed fill color
        color = "white",     # Border color
        weight = 1,           # Border thickness
        opacity = 0.8,        # Border opacity
        fillOpacity = input$opacity,    # Polygon fill opacity
        popup = ~paste(
          "<strong>District:</strong>", DIST_NUM
        )
      ) %>%
      addCircleMarkers(
        data = filtered_data,
        lng = ~Longitude, lat = ~Latitude,
        color = "red",
        weight = 2,
        opacity = 1,  
        radius = 4,
        popup = ~paste(
          "<strong>Crime:</strong>", `Primary Type`, "<br>",
          "<strong>Description:</strong>", Description, "<br>",
          "<strong>Block:</strong>", Block, "<br>",
          "<strong>District:</strong>", Districts, "<br>",
          "<strong>Date:</strong>", Date
        )
      ) %>%
      setView(lng = -87.6298, lat = 41.8781, zoom = 10) # Center the map on Chicago
  })
  
  # Render heatmap
  output$heatmap <- renderLeaflet({
    police_district_boundaries <- police_district_boundaries %>%
      mutate(
        Count = if (input$year_heatmap == "2022") {
          Count_2022
        } else if (input$year_heatmap == "2023") {
          Count_2023
        } else {
          Total_Count
        }
      ) %>%
      filter(Count >= input$crime_range[1] & Count <= input$crime_range[2])
    
    leaflet(data = police_district_boundaries) %>%
      addTiles() %>%
      addPolygons(
        fillColor = ~colorNumeric("YlOrRd", Count)(Count),
        color = "white",
        weight = 1,
        opacity = 0.8,
        fillOpacity = 0.7,
        popup = ~paste(
          "<strong>District:</strong>", DIST_NUM, "<br>",
          "<strong>Reported Crimes:</strong>", Count
        )
      ) %>%
      setView(lng = -87.6298, lat = 41.8781, zoom = 10)
  })
}

# Run the Shiny app
shinyApp(ui, server)
