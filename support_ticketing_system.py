# expected file:
# - topic mapping with index in CSV file 
# - tfidf_vectorizer in pickle file
# - trained machine learning model in pickle file

# import packages
import datetime
import pandas as pd

from deploy_backend import load_files, get_ticket_category_and_priority

import altair as alt
import streamlit as st
from streamlit import session_state as ss

# overwrite the current DataFrame with live-updated ticket status 
def save_edits():
    # concatenating the ticket_df and edited_ticket_df merges the changes on the edited_ticket_df on ticket_df
    # this will result in duplicated ticket IDs with different ticket status
    # the concat_df will contains all records from ticket_df, then all records from edited_ticket_df
    concat_df = pd.concat([ss.ticket_df,ss.edited_ticket_df.copy()],axis=0)
    # if duplicated Ticket ID exists for a ticket, then the ticket_status in the edited_ticket_df will be kept
    concat_df.drop_duplicates(subset='ID',keep='last',ignore_index=True,inplace=True)
    # updating both ticket_df and edited_ticket_df with the concat_df with merged changes
    ss.ticket_df = concat_df
    ss.edited_ticket_df = concat_df

def submit_ticket_page(topic_df, vectorizer, classifier):
    # Show a section to add a new ticket.
    st.header("Submit a ticket")

    # We're adding tickets via an `st.form` and some input widgets. If widgets are used
    # in a form, the app will only rerun once the submit button is pressed.
    with st.form("add_ticket_form"):
        title = st.text_input("Title")
        issue = st.text_area("Describe the issue") # enter description [long_text]
        submitted = st.form_submit_button("Submit")

    if submitted:
      # Make a dataframe for the new ticket and append it to the dataframe in session state.
        try:
          recent_ticket_number = int(max(ss.ticket_df.ID).split("-")[1])
        except ValueError:
          recent_ticket_number = 0
        category,priority = get_ticket_category_and_priority(issue, topic_df, vectorizer, classifier)
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        # df add new row at the top of current ticket_df
        df_new = pd.DataFrame(
          [
              {
                "ID": f"TICKET-{recent_ticket_number+1}", # TICKET-{id_in_int}
                "Title":title,
                "Status":"Open",
                "Description":issue,
                "Category":category,
                "Priority":priority,
                "Date Submitted":today
              }
          ]
        )
        # Show a little success message.
        st.write("Ticket submitted! Here are the ticket details:")
        st.table(df_new.astype(str).rename(index={0:'Information'}).T) # , use_container_width=True, hide_index=True
        ss.ticket_df = pd.concat([df_new, ss.ticket_df], axis=0)
    return

def view_all_tickets_page():
    ss.ticket_df = ss.ticket_df.copy()
    # Show section to view and edit existing tickets in a table.
    st.header("Existing tickets")
    st.write(f"Number of tickets: `{len(ss.ticket_df)}`")

    st.info(
        "You can edit the tickets by double clicking on a cell. Note how the plots below "
        "update automatically! You can also sort the table by clicking on the column headers.",
        icon="✍️",
    )

    # Show the tickets dataframe with `st.data_editor`. This lets the user edit the table
    # cells. The edited data is returned as a new dataframe.
    ss.edited_ticket_df = st.data_editor(
        ss.ticket_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Status": st.column_config.SelectboxColumn(
                "Status",
                help="Ticket status",
                options=["Open", "In Progress", "Closed"],
                required=True,
            )
        },
        # Disable editing the ID and Date Submitted columns.
        disabled=["ID", "Title","Description","Category", "Priority","Date Submitted"],
        on_change = save_edits
    )
    st.write("edited_ticket_df")
    st.dataframe(ss.edited_ticket_df)
    return

def ticket_dashboard_page():
    # Show some metrics and charts about the ticket.
    st.header("Statistics")

    # Show metrics side by side using `st.columns` and `st.metric`.
    col1, col2, col3 = st.columns(3)
    col1.metric(
       label="Number of OPEN tickets", 
       value=len(ss.edited_ticket_df[ss.edited_ticket_df.Status == "Open"])
       )
    col2.metric(
       label="Number of IN PROGRESS tickets", 
       value=len(ss.edited_ticket_df[ss.edited_ticket_df.Status == "In Progress"])
       )
    col3.metric(
       label="Number of CLOSED tickets", 
       value=len(ss.edited_ticket_df[ss.edited_ticket_df.Status == "Closed"])
       )

    # Show two Altair charts using `st.altair_chart`.
    st.write("")
    st.write("##### Ticket status per month")
    status_plot = (
        alt.Chart(ss.edited_ticket_df)
        .mark_bar()
        .encode(
            x="month(Date Submitted):O",
            y="count():Q",
            xOffset="Status:N",
            color="Status:N",
        )
        .configure_legend(
            orient="bottom", titleFontSize=14, labelFontSize=14, titlePadding=5
        )
    )
    st.altair_chart(status_plot, use_container_width=True, theme="streamlit")

    st.write("##### Current ticket priorities")
    priority_plot = (
        alt.Chart(ss.edited_ticket_df)
        .mark_arc()
        .encode(theta="count():Q", color="Priority:N")
        .properties(height=300)
        .configure_legend(
            orient="bottom", titleFontSize=14, labelFontSize=14, titlePadding=5
        )
    )
    st.altair_chart(priority_plot, use_container_width=True, theme="streamlit")

    st.write("##### Ticket category per month")
    status_plot = (
        alt.Chart(ss.ticket_df)
        .mark_bar()
        .encode(
            x="month(Date Submitted):O",
            y="count():Q",
            xOffset="Category:N",
            color="Category:N",
        )
        .configure_legend(
            orient="bottom", titleFontSize=14, labelFontSize=14, titlePadding=5
        )
    )
    st.altair_chart(status_plot, use_container_width=True, theme="streamlit")
    return

def main():
    topic_df, vectorizer, classifier = load_files()

    # Financial Domain Complaint Ticketing System
    st.set_page_config(page_title="Financial Domain Complaint Ticketing System", page_icon="🎫")
    st.title("🎫 Financial Domain Complaint Ticketing System")
    st.write(
        """
        This app shows how you can build an internal tool in Streamlit. Here, we are 
        implementing a support ticket workflow. The user can create a ticket, edit 
        existing tickets, and view some statistics.
        """
    )

    if "ticket_df" not in ss:
        # Create a Pandas dataframe to store tickets.
        ss.ticket_df = pd.DataFrame({
            "ID": [], # TICKET-{id_in_int}
            "Title":[],
            "Status":[],
            "Description":[],
            "Category":[],
            "Priority":[],
            "Date Submitted":[]
        })
        ss.edited_ticket_df = ss.ticket_df.copy()

    # Sidebar to select page and commit changes upon selection
    page = st.sidebar.selectbox("Select: ", ("Submit a Ticket","View all Tickets"), on_change=save_edits) # , on_change=save_edits

    if page == "Submit a Ticket":
        submit_ticket_page(topic_df, vectorizer, classifier)
    elif page == "View all Tickets":
        view_all_tickets_page()
        ticket_dashboard_page()
     
if __name__ == "__main__":
   main()