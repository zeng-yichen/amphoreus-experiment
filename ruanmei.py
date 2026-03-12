import argparse
from post import generate_iterative_linkedin_posts
from brief import generate_briefing

def main():
    parser = argparse.ArgumentParser(description="Ruan Mei Content Generation Engine")
    parser.add_argument(
        "action", 
        choices=["post", "brief"], 
        help="Type 'post' to generate LinkedIn posts, or 'brief' to generate a ghostwriter briefing."
    )
    parser.add_argument(
        "client_name", 
        type=str, 
        help="The full name of the client (e.g., 'Ruan Mei')"
    )
    parser.add_argument(
        "company_keyword", 
        type=str, 
        help="The directory name for the client's files (e.g., 'Genius Society')"
    )

    args = parser.parse_args()

    if args.action == "post":
        print(f"Ruan Mei: I'm Starting Post Generation for {args.client_name}...")
        generate_iterative_linkedin_posts(args.client_name, args.company_keyword)
    elif args.action == "brief":
        print(f"Ruan Mei: I'm Starting Briefing for {args.client_name}...")
        generate_briefing(args.client_name, args.company_keyword)

if __name__ == "__main__":
    main()