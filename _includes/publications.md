<h2 id="publications" style="margin: 2px 0px -15px;">Publications</h2>

<div class="publications">
<ol class="bibliography">

{% for link in site.data.publications.main %}

<li>
<div class="pub-row">
  <!-- Publication details container -->
  <div class="col-sm-9" style="position: relative;padding-right: 15px;padding-left: 20px;">
      <!-- Publication title -->
      <div class="title"><a href="{{ link.pdf }}">{{ link.title }}</a></div>
      <!-- Authors of the publication -->
      <div class="author">{{ link.authors }}</div>
      <!-- Conference or journal name -->
      <div class="periodical"><em>{{ link.conference }}</em></div>
    <!-- Links to additional resources -->
    <div class="links">
      {% if link.pdf %} 
      <!-- Link to PDF -->
      <a href="{{ link.pdf }}" class="btn btn-sm z-depth-0" role="button" target="_blank" style="font-size:12px;">PDF</a>
      {% endif %}
      {% if link.code %} 
      <!-- Link to code repository -->
      <a href="{{ link.code }}" class="btn btn-sm z-depth-0" role="button" target="_blank" style="font-size:12px;">Code</a>
      {% endif %}
      {% if link.page %} 
      <!-- Link to project page -->
      <a href="{{ link.page }}" class="btn btn-sm z-depth-0" role="button" target="_blank" style="font-size:12px;">Project Page</a>
      {% endif %}
      {% if link.bibtex %} 
      <!-- Link to BibTex entry -->
      <a href="{{ link.bibtex }}" class="btn btn-sm z-depth-0" role="button" target="_blank" style="font-size:12px;">BibTex</a>
      {% endif %}
      {% if link.notes %} 
      <!-- Additional notes -->
      <strong><i style="color:#e74d3c">{{ link.notes }}</i></strong>
      {% endif %}
      {% if link.others %} 
      <!-- Other information -->
      {{ link.others }}
      {% endif %}
    </div>
  </div>
</div>
</li>

{% endfor %}

</ol>
</div>
