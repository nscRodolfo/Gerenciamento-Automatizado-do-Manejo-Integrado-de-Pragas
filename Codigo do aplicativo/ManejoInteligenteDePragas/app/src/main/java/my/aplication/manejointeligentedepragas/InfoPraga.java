package my.aplication.manejointeligentedepragas;

import android.content.Intent;
import android.content.SharedPreferences;

//import android.support.v4.content.ContextCompat;
import androidx.core.content.ContextCompat;
//import android.support.v4.view.ViewPager;
import androidx.viewpager.widget.ViewPager;

//import android.support.annotation.NonNull;
import androidx.annotation.NonNull;
//import android.support.design.widget.NavigationView;
import com.google.android.material.navigation.NavigationView;
//import android.support.v4.view.GravityCompat;
import androidx.core.view.GravityCompat;
//import android.support.v4.widget.DrawerLayout;
import androidx.drawerlayout.widget.DrawerLayout;
//import android.support.v7.app.ActionBarDrawerToggle;
import androidx.appcompat.app.ActionBarDrawerToggle;
//import android.support.v7.app.AppCompatActivity;
import androidx.appcompat.app.AppCompatActivity;
//import android.support.v7.widget.Toolbar;
import androidx.appcompat.widget.Toolbar;

import android.os.Bundle;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import my.aplication.manejointeligentedepragas.Auxiliar.Utils;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_Usuario;
import my.aplication.manejointeligentedepragas.ImageAdapter.ViewPagerAdapter;

import com.example.manejointeligentedepragas.R;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;

public class InfoPraga extends AppCompatActivity implements NavigationView.OnNavigationItemSelectedListener{

    int codPraga;

    TextView tvDescricaoBioecologia;
    TextView tvInjurias;
    TextView tvNome;
    TextView tvFamilia;
    TextView tvOrdem;
    TextView tvNomeCientifico;
    TextView tvLocalizacaoAmostra;
    TextView tvAmbientePropicio;
    TextView tvCicloVida;
    TextView tvObservacoes;
    TextView tvHorarioAtuacao;
    TextView tvEstagioDeAtuacao;
    TextView tvControleCultural;
    Button btnInfoPragaMetodo;
    Button btnInfoPragaInimigo;
    LinearLayout sliderDotspanel;

    //pontos indicadores
    private int dotscount;
    private ImageView[] dots;


    String nome;
    String familia;
    String ordem;
    String descricao_bioecologia;
    String nomeCientificoEspecie;
    String localizacao_amostra;
    String ambientePropicio;
    String cicloVida;
    String injurias;
    String observacoes;
    String horarioAtuacao;
    String estagioDeAtuacao;
    String controleCultural;

    ArrayList<String> urlsPragas = new ArrayList<>();

    private DrawerLayout drawerLayout;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_info_praga);

        tvDescricaoBioecologia = findViewById(R.id.Descricao);
        tvInjurias = findViewById(R.id.Injurias);
        tvNome = findViewById(R.id.InfoPragaNome);
        tvFamilia = findViewById(R.id.InfoPragaFamilia);
        tvOrdem = findViewById(R.id.InfoPragaOrdem);
        tvNomeCientifico = findViewById(R.id.InfoPragaNomeCientifico);
        tvAmbientePropicio = findViewById(R.id.InfoPragaAmbientePropicio);
        tvLocalizacaoAmostra = findViewById(R.id.InfoPragaAmostra);
        tvCicloVida = findViewById(R.id.InfoPragaCicloDeVida);
        tvObservacoes = findViewById(R.id.InfoPragaObservacoes);
        tvHorarioAtuacao = findViewById(R.id.InfoPragaHorarioDeAtuacao);
        tvEstagioDeAtuacao = findViewById(R.id.InfoPragaEstagioDeAtuacao);
        tvControleCultural = findViewById(R.id.InfoPragaControleCultural);
        btnInfoPragaMetodo = findViewById(R.id.btnInfoPragaMetodo);
        btnInfoPragaInimigo = findViewById(R.id.btnInfoPragaInimigo);
        ViewPager viewPager = findViewById(R.id.ViewPagerPraga);
        sliderDotspanel = findViewById(R.id.SliderDots);


        //menu novo
        Toolbar toolbar = findViewById(R.id.toolbar_infoPraga);
        setSupportActionBar(toolbar);
        drawerLayout= findViewById(R.id.drawer_layout_infoPraga);
        NavigationView navigationView = findViewById(R.id.nav_view_infoPraga);
        navigationView.setNavigationItemSelectedListener(this);
        ActionBarDrawerToggle toggle = new ActionBarDrawerToggle(this, drawerLayout, toolbar, R.string.navigation_drawer_open, R.string.navigation_drawer_close);
        drawerLayout.addDrawerListener(toggle);
        toggle.syncState();
        View headerView = navigationView.getHeaderView(0);

        Controller_Usuario controller_usuario = new Controller_Usuario(getBaseContext());
        String nomeUsu = controller_usuario.getUser().getNome();
        String emailUsu = controller_usuario.getUser().getEmail();

        TextView nomeMenu = headerView.findViewById(R.id.nomeMenu);
        nomeMenu.setText(nomeUsu);

        TextView emailMenu = headerView.findViewById(R.id.emailMenu);
        emailMenu.setText(emailUsu);


        codPraga = getIntent().getIntExtra("Cod_Praga",0);

        ResgatarUrlPragas(viewPager, codPraga);
        ResgataPragas(codPraga);

        viewPager.addOnPageChangeListener(new ViewPager.OnPageChangeListener() {
            @Override
            public void onPageScrolled(int position, float positionOffset, int positionOffsetPixels) {

            }

            @Override
            public void onPageSelected(int position) {

                for(int i = 0; i< dotscount; i++){
                    dots[i].setImageDrawable(ContextCompat.getDrawable(getApplicationContext(), R.drawable.non_active_dot));
                }

                dots[position].setImageDrawable(ContextCompat.getDrawable(getApplicationContext(), R.drawable.active_dot));
            }

            @Override
            public void onPageScrollStateChanged(int state) {

            }
        });


        btnInfoPragaMetodo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i= new Intent(InfoPraga.this,VisualizaMetodos.class);
                i.putExtra("Cod_Praga",codPraga);
                startActivity(i);
            }
        });

        btnInfoPragaInimigo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i= new Intent(InfoPraga.this,VisualizaInimigos.class);
                i.putExtra("Cod_Praga",codPraga);
                startActivity(i);
            }
        });


    }

    @Override
    public void onBackPressed() {
        if(drawerLayout.isDrawerOpen(GravityCompat.START)){
            drawerLayout.closeDrawer(GravityCompat.START);
        }else {
            super.onBackPressed();
        }
    }

    @Override
    public boolean onNavigationItemSelected(@NonNull MenuItem menuItem) {
        switch (menuItem.getItemId()){
            case R.id.drawerPerfil:
                Intent i= new Intent(this, Perfil.class);
                startActivity(i);
                break;
            case R.id.drawerProp:
                Intent prop= new Intent(this, Propriedades.class);
                startActivity(prop);
                break;

            case R.id.drawerPlantas:
                Intent j = new Intent(this, VisualizaPlantas.class);
                startActivity(j);
                break;

            case R.id.drawerPrag:
                Intent k = new Intent(this, VisualizaPragas.class);
                startActivity(k);
                break;

            case R.id.drawerMet:
                Intent l = new Intent(this, VisualizaMetodos.class);
                startActivity(l);
                break;

            case R.id.drawerSobreMip:
                Intent p = new Intent(this, SobreMIP.class);
                startActivity(p);
                break;

            case R.id.drawerTutorial:
                SharedPreferences pref = getApplicationContext().getSharedPreferences("myPrefs",MODE_PRIVATE);
                SharedPreferences.Editor editor = pref.edit();
                editor.putBoolean("isIntroOpened",false);
                editor.commit();

                Intent intro = new Intent(this, IntroActivity.class);
                startActivity(intro);
                break;

            case R.id.drawerSobre:
                Intent pp = new Intent(this, Sobre.class);
                startActivity(pp);
                break;

            case R.id.drawerReferencias:
                Intent pi = new Intent(this, Referencias.class);
                startActivity(pi);
                break;

            case R.id.drawerRecomendações:
                Intent pa = new Intent(this, RecomendacoesMAPA.class);
                startActivity(pa);
                break;
        }
        drawerLayout.closeDrawer(GravityCompat.START);
        return true;
    }


    public void ResgataPragas(int codP){
        Utils u = new Utils();
        if(!u.isConected(getBaseContext()))
        {
            Toast.makeText(this,"Habilite a conexão com a internet!", Toast.LENGTH_LONG).show();
        }else { // se tem acesso à internet
            String url = "https://mip.software/phpapp/infoPraga.php?Cod_Praga="+codP;

            RequestQueue queue = Volley.newRequestQueue(this);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    //Parsing json
                    //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                    try {
                        //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                        JSONArray array = new JSONArray(response);
                        for (int i = 0; i< array.length(); i++){
                            JSONObject obj = array.getJSONObject(i);

                            nome = obj.getString("Nome");
                            familia = obj.getString("Familia");
                            ordem = obj.getString("Ordem");
                            nomeCientificoEspecie = obj.getString("NomeCientifico");
                            localizacao_amostra = obj.getString("Localizacao");
                            ambientePropicio = obj.getString("AmbientePropicio");
                            cicloVida = obj.getString("CicloVida");
                            observacoes = obj.getString("Observacoes");
                            horarioAtuacao = obj.getString("HorarioDeAtuacao");
                            estagioDeAtuacao = obj.getString("EstagioDeAtuacao");
                            injurias = obj.getString("Injurias");
                            descricao_bioecologia = obj.getString("Descricao");
                            controleCultural = obj.getString("ControleCultural");
                        }

                        tvDescricaoBioecologia.setText("Bioecologia: "+descricao_bioecologia);
                        tvInjurias.setText("Injúrias: "+injurias);
                         tvNome.setText("Nome: "+nome);
                         tvFamilia.setText("Família: "+familia);
                         tvOrdem.setText("Ordem: "+ordem);
                         tvNomeCientifico.setText("Nome científico: "+nomeCientificoEspecie);
                         tvLocalizacaoAmostra.setText("Amostra: "+localizacao_amostra);
                         tvAmbientePropicio.setText("Ambiente propício: "+ambientePropicio);
                         tvCicloVida.setText("Ciclo de vida: "+cicloVida);
                         tvObservacoes.setText("Observações: "+observacoes);
                         tvHorarioAtuacao.setText("Horário de atuação: "+horarioAtuacao);
                         tvEstagioDeAtuacao.setText("Estágio de atuação: "+estagioDeAtuacao);
                         tvControleCultural.setText("Controles culturais: "+controleCultural);

                        setTitle("MIP² | "+nome);



                    } catch (JSONException e) {
                        Toast.makeText(InfoPraga.this, e.toString(), Toast.LENGTH_LONG).show();
                    }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    Toast.makeText(InfoPraga.this,error.toString(), Toast.LENGTH_LONG).show();
                }
            }));

        }
    }

    public void ResgatarUrlPragas(final ViewPager viewPager, int codP){
        Utils u = new Utils();
        if (!u.isConected(getBaseContext())) {
            Toast.makeText(this, "Habilite a conexão com a internet", Toast.LENGTH_LONG).show();
        } else { // se tem acesso à internet

            String url = "https://mip.software/phpapp/resgatarFotoPragas.php?Cod_Praga="+codP;

            RequestQueue queue = Volley.newRequestQueue(InfoPraga.this);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    //Parsing json
                    //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                    try {
                        JSONArray array = new JSONArray(response);
                        for (int i = 0; i < array.length(); i++) {
                            JSONObject obj = array.getJSONObject(i);
                            urlsPragas.add("https://mip.software/pragas/"+obj.getString("FotoPraga"));
                        }
                        ViewPagerAdapter adapterPragas = new ViewPagerAdapter(InfoPraga.this,urlsPragas);
                        viewPager.setAdapter(adapterPragas);

                        dotscount = adapterPragas.getCount();
                        dots = new ImageView[dotscount];


                        for(int i = 0; i < dotscount; i++){

                            dots[i] = new ImageView(InfoPraga.this);
                            dots[i].setImageDrawable(ContextCompat.getDrawable(getApplicationContext(), R.drawable.non_active_dot));

                            LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT);

                            params.setMargins(8, 0, 8, 0);

                            sliderDotspanel.addView(dots[i], params);

                        }

                        dots[0].setImageDrawable(ContextCompat.getDrawable(getApplicationContext(), R.drawable.active_dot));

                    } catch (JSONException e) {
                        Toast.makeText(InfoPraga.this, e.toString(), Toast.LENGTH_LONG).show();
                    }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    Toast.makeText(InfoPraga.this, error.toString(), Toast.LENGTH_LONG).show();
                }
            }));
        }

    }

}
