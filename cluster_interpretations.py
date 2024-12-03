def get_cluster_interpretations():
    return {
        "title": "Major Findings",
        "clusters": {
            0: {
                "name": "Low-Engagement Customers",
                "percentage": "27.78%",
                "description": "Represents a significant portion of customers who show the highest website visit distribution but minimal campaign engagement. Despite their active browsing behavior, they demonstrate consistently low distributions in campaign acceptance, spending, and CVR (around 1-2%). Their income distribution is among the lowest, which may explain their limited purchasing power. This presents a clear opportunity for the company to better understand why these frequent browsers rarely convert and develop strategies to capture this substantial segment's potential.",
                "recommendations": {
                    "Friction Point Analysis & Resolution": "Given their high website visits but low conversion, conduct detailed user journey analysis to identify and eliminate conversion barriers. This may include implementing A/B testing on landing pages, optimizing the checkout process, and analyzing exit points to understand abandonment reasons.",
                    "Value-Based Pricing Strategy": "Develop entry-level product offerings or payment plans that align with their lower income level. Consider introducing tiered pricing, starter packages, or special promotions that make products more accessible while maintaining value perception."
                }
            },
            1: {
                "name": "Mid-Tier Active Customers" ,
                "percentage": "27.19%",
                "description": "Emerges as the highest-value segment with exceptional performance metrics. The boxplots reveal this cluster has the highest distributions in campaign acceptance, income levels, total spending, and CVR (ranging from 5-15% with outliers reaching above 20%). Interestingly, despite having the lowest web visit frequency, their strong engagement with campaigns and high conversion rates make them the most valuable customer segment. This suggests highly effective campaign targeting and relevance for this high-income group, indicating potential for even greater returns through personalized premium offerings.",
                "recommendations": {
                    "Premium Experience Enhancement": "Develop an exclusive shopping experience for this high-value segment. This could include priority customer service, early access to new products, and VIP events, leveraging their high income and strong buying intent.",
                    "Advanced Personalization": "Implement sophisticated personalization strategies using their robust purchase history and campaign response data. Create predictive models to anticipate their needs and preferences, enabling proactive product recommendations and tailored campaign messaging."
                }
            },
            2: {
                "name": "Premium Engaged Customers",
                "percentage": "28.37%",
                "description": "Shows promising balanced behavior across metrics. The boxplots indicate high web visit distributions, moderate campaign acceptance, and steady CVR (around 2-3%). Their income distribution sits comfortably in the middle range, resulting in consistent purchase frequency and spending patterns, though not as high as Cluster 1. As the largest segment, this group presents a significant opportunity for upward migration through targeted engagement strategies.",
                "recommendations": {
                    "Strategic Upgrade Path": "Design a clear pathway to migrate these customers toward premium spending levels. Implement a tiered rewards program that incentivizes increased purchase frequency and spending, with visible benefits at each tier.",
                    "Engagement Optimization": "Leverage their moderate campaign responsiveness and high browsing behavior to create more targeted engagement strategies. Use behavioral data to optimize campaign timing and messaging, focusing on their demonstrated interests and purchase patterns."
                }
            },
            3: {
                "name": "High-Browse Low-Convert",
                "percentage": "16.66%",
                "description": "Is the smallest segment showing consistently low performance across all metrics. The boxplots demonstrate low distributions in web visits, campaign acceptance, and CVR (similar to Cluster 0), despite having a slightly higher income distribution than Cluster 0. Their limited engagement across all touchpoints suggests a need for re-engagement strategies or potential evaluation of their fit with the company's target market.",
                "recommendations": {
                    "Re-engagement Campaign": "Launch a focused re-engagement program targeting this segment's specific pain points. This could include conducting surveys to understand their low engagement, offering win-back promotions, and creating simplified pathways to purchase.",
                    "Customer Journey Evaluation": "Analyze the full customer journey for this segment to identify where and why engagement drops off. Consider implementing triggered communications at key touchpoints and simplified conversion paths to reduce barriers to purchase."
                }
            }
        },
        "cross_cluster_initiatives": {
            "Segmented Marketing Automation": "Develop automated marketing workflows tailored to each cluster's characteristics, ensuring messaging and offers align with their browsing patterns, income levels, and historical response rates.",
            "Data-Driven Personalization": "Implement a robust personalization engine that adapts content, pricing, and promotions based on cluster membership and individual behavior patterns, with specific focus on moving customers up the value chain."
        }
    } 